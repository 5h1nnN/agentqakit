"""Paired, budgeted coding-agent trial using real DeepSeek and MCP tool calls.

Only synthetic task code is transmitted. Credentials are loaded in the host and
are never passed to candidate processes, MCP subprocesses, prompts or outputs.
The host is deliberately bounded; results do not benchmark a commercial IDE.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from openai import AsyncOpenAI

from agentqa.evaluators.deepseek import load_llm_environment
from repair_fixture import TASK_IDS, make_workspace, evaluate_workspace

ROOT=Path(__file__).resolve().parents[3]
PYTHON=Path(sys.executable).resolve()
MAX_ROUNDS=12
MAX_TOKENS=60_000
MAX_SECONDS=240
MAX_COMPLETION=4096
MAX_TOOL_CALLS_PER_ROUND=5
MCP_NAMES={'validate_tests','run_tests','get_run_result','compare_runs','evaluate_gate'}

def dump(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def digest_file(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def log(path:Path,item:dict)->None:
    with path.open('a',encoding='utf-8') as out:
        out.write(json.dumps(item,ensure_ascii=False)+'\n')

def fn(name:str,description:str,properties:dict,required:list[str])->dict:
    return {'type':'function','function':{'name':name,'description':description,
        'parameters':{'type':'object','properties':properties,'required':required,
                      'additionalProperties':False}}}

COMMON_TOOLS=[
 fn('list_files','List public files inside this task workspace.',{},[]),
 fn('read_file','Read a public workspace-relative UTF-8 file, at most 24000 characters.',
    {'path':{'type':'string'}},['path']),
 fn('edit_file','Replace an exact unique text span in an editable source file.',
    {'path':{'type':'string'},'old_text':{'type':'string'},'new_text':{'type':'string'}},
    ['path','old_text','new_text']),
 fn('write_test','Write a custom test under tests/test_*.py; cannot edit supplied specifications.',
    {'path':{'type':'string'},'content':{'type':'string'}},['path','content']),
 fn('run_python','Run a short local Python check against campus_app, services, or tests. '
    'Imports allowed: campus_app, services, asyncio, json, math, time, unittest, pytest. '
    'No filesystem/network/process APIs. 8 second limit. Example: '
    'from campus_app import make_agent\na=make_agent()\nprint(a.invoke({"action":"query_course","course_id":"AI101"}))',
    {'code':{'type':'string'}},['code']),
 fn('finish','Submit an assessment after actual edits/checks. release_ready is your final release recommendation. '
    'Do not claim checks you did not perform.',
    {'release_ready':{'type':'boolean'},'defect_found':{'type':'boolean'},
     'summary':{'type':'string'},'evidence':{'type':'string'}},
    ['release_ready','defect_found','summary','evidence'])
]

def child_env(workspace:Path)->dict[str,str]:
    # No API tokens, home directories, config paths or parent PYTHONPATH.
    keep={k:v for k,v in os.environ.items() if k.upper() in
          {'SYSTEMROOT','WINDIR','PATH','TEMP','TMP','COMSPEC','PATHEXT'}}
    keep.update(PYTHONPATH=str(workspace),PYTHONDONTWRITEBYTECODE='1',
                PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    return keep

class TaskTools:
    def __init__(self,workspace:Path,editable:list[str]):
        self.workspace=workspace.resolve()
        self.editable=set(editable)
        self.usage={'edits':0,'python_calls':0,'mcp_calls':0,'local_test_timeouts':0}

    def resolve(self,relative:str)->Path:
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts or '\x00' in relative:
            raise ValueError('Only workspace-relative paths are allowed')
        resolved=(self.workspace/p).resolve()
        resolved.relative_to(self.workspace)
        return resolved

    def python(self,code:str)->dict:
        if len(code)>20000:raise ValueError('test code too large')
        tree=ast.parse(code)
        allowed={'campus_app','services','asyncio','json','math','time','unittest','pytest'}
        forbidden={'open','exec','eval','compile','__import__','globals','locals','vars','input','breakpoint'}
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=([node.module or ''] if isinstance(node,ast.ImportFrom)
                       else [a.name for a in node.names])
                if any(n.split('.')[0] not in allowed for n in names):
                    raise ValueError('Import is outside test tool allowlist')
            if isinstance(node,ast.Name) and node.id in forbidden:
                raise ValueError('This runtime API is outside test tool allowlist')
            if isinstance(node,ast.Attribute) and node.attr.startswith('__'):
                raise ValueError('Dunder introspection is not exposed')
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                if '../' in node.value or '..\\' in node.value or ':\\' in node.value:
                    raise ValueError('Outside paths are not exposed')
        self.usage['python_calls']+=1
        try:
            r=subprocess.run([str(PYTHON),'-B','-c',code],cwd=self.workspace,
                env=child_env(self.workspace),capture_output=True,text=True,encoding='utf-8',
                errors='replace',timeout=8)
            return {'exit_code':r.returncode,'stdout':r.stdout[:16000],'stderr':r.stderr[:8000]}
        except subprocess.TimeoutExpired as exc:
            self.usage['local_test_timeouts']+=1
            return {'exit_code':None,'timeout_seconds':8,'stdout':str(exc.stdout or '')[:8000]}

    async def call(self,name:str,args:dict)->dict:
        if name=='list_files':
            return {'files':sorted(str(p.relative_to(self.workspace)).replace('\\','/')
                for p in self.workspace.rglob('*') if p.is_file() and
                '__pycache__' not in p.parts and '.agentqa' not in p.parts)[:100]}
        if name=='read_file':
            p=self.resolve(args['path'])
            if p.stat().st_size>500000:raise ValueError('file is too large')
            return {'path':args['path'],'content':p.read_text(encoding='utf-8')[:24000]}
        if name=='edit_file':
            rel=args['path'].replace('\\','/')
            if rel not in self.editable:raise ValueError('Supplied infrastructure/specs are read-only')
            p=self.resolve(rel);text=p.read_text(encoding='utf-8')
            old=args['old_text'];new=args['new_text']
            if not old or text.count(old)!=1:raise ValueError('old_text must match exactly once')
            modified=text.replace(old,new,1)
            parsed=ast.parse(modified)
            # Candidate code has no reason to access files, host configuration,
            # processes or network in these pure business-logic tasks.
            safe_imports={'services','typing','collections','copy','re','json','math','asyncio','dataclasses'}
            for node in ast.walk(parsed):
                if isinstance(node,(ast.Import,ast.ImportFrom)):
                    names=([node.module or ''] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names])
                    if any(n.split('.')[0] not in safe_imports for n in names):
                        raise ValueError('Business source imports must stay within task libraries')
                if isinstance(node,ast.Name) and node.id in {'open','exec','eval','compile','__import__','globals','locals'}:
                    raise ValueError('Host filesystem/process APIs are not part of this task')
                if isinstance(node,ast.Attribute) and node.attr.startswith('__'):
                    raise ValueError('Dunder introspection is outside task scope')
            p.write_text(modified,encoding='utf-8')
            self.usage['edits']+=1
            return {'edited':rel,'sha256':digest_file(p)}
        if name=='write_test':
            rel=args['path'].replace('\\','/')
            if not rel.startswith('tests/test_') or not rel.endswith('.py'):
                raise ValueError('Custom tests must be tests/test_*.py')
            ast.parse(args['content']);p=self.resolve(rel)
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(args['content'],encoding='utf-8')
            return {'written':rel}
        if name=='run_python':return await asyncio.to_thread(self.python,args['code'])
        raise ValueError('unknown tool')

def resource_text(result:Any)->str:
    return '\n'.join(getattr(part,'text','') for part in result.contents)

async def mcp_description(client:Client)->list[dict]:
    listing=await client.list_tools()
    tools=[]
    for item in listing.tools:
        if item.name not in MCP_NAMES:continue
        schema=getattr(item,'input_schema',None) or item.model_dump(by_alias=True).get('inputSchema')
        tools.append({'type':'function','function':{
            'name':item.name,'description':item.description or '', 'parameters':schema}})
    tools.append(fn('read_resource','Read a bounded AgentQAKit resource URI returned by its tools.',
                    {'uri':{'type':'string'}},['uri']))
    return tools

async def mcp_call(client:Client,name:str,args:dict)->dict:
    if name=='read_resource':
        uri=args['uri']
        if not uri.startswith('agentqa://'):raise ValueError('Only AgentQAKit resources are available')
        return {'uri':uri,'content':resource_text(await client.read_resource(uri))[:28000]}
    if name not in MCP_NAMES:raise ValueError('This MCP tool is not exposed for the trial')
    r=await client.call_tool(name,args)
    value=getattr(r,'structured_content',None)
    return {'is_error':bool(getattr(r,'is_error',False)),
            'result':value if value is not None else [getattr(x,'text','') for x in r.content]}

def safe_exception(exc:BaseException)->str:
    # Never persist HTTP request objects or credential-bearing SDK reprs.
    return type(exc).__name__

async def trial(output:Path,task_id:str,arm:str,repeat:int,client:AsyncOpenAI,
                model:str,semaphore:asyncio.Semaphore)->dict:
    async with semaphore:
        case=f'{task_id}-{arm}-r{repeat}'
        folder=output/'trials'/case
        if folder.exists():raise RuntimeError('Never overwrite or silently resume a trial')
        workspace=folder/'workspace';folder.mkdir(parents=True)
        info=make_workspace(task_id,workspace)
        dump(folder/'fixture.json',info)
        pre=await evaluate_workspace(workspace)
        dump(folder/'before-oracle.json',pre)
        tools_host=TaskTools(workspace,info['editable_files'])
        timeline=folder/'events.jsonl'
        usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0,
               'prompt_cache_hit_tokens':0,'prompt_cache_miss_tokens':0}
        started=time.monotonic();api_calls=0;finish=None;status='running';provider=[]
        sources_before={n:(workspace/n).read_text(encoding='utf-8') for n in info['editable_files']}
        context=info['public_requirements']
        system=('You are a coding agent auditing and repairing a small campus service. '
            'Use only the exposed tools and this workspace. Review all stated business requirements. '
            'Actually edit code when necessary and verify it with tools; preserve legitimate functionality. '
            'You do not know whether this snapshot contains any defect. Never assume that it does. '
            'Do not invent executed checks. Supplied infrastructure and test specifications are read-only. '
            'Only editable business source and custom tests may be changed. '
            'These pure task modules may not add filesystem, network or process APIs. '
            'Inspect relevant files and use the available testing tools efficiently. '
            'Finish with the finish tool giving a release recommendation and concise evidence. '
            f'Budget: {MAX_ROUNDS} model turns, {MAX_SECONDS} seconds, {MAX_TOKENS} cumulative tokens. '
            'At most five tool calls are accepted per turn. Respond in concise Chinese or English.')
        prompt=('请审查这个校园事务 Agent 的实现是否符合以下公开要求，必要时真正修复，'
            '并提交是否可发布的建议。不要只讨论方案。\n\n'+context+
            '\n\nEditable files: '+json.dumps(info['editable_files'])+
            '\nCommon entry: from campus_app import make_agent; a=make_agent(); a.invoke(payload). '
            'The services module contains the read-only dependency simulator. '
            'Public AgentTest specifications are also readable by both arms. '
            'Use custom Python/unittest/pytest checks as needed.')
        messages=[{'role':'system','content':system},{'role':'user','content':prompt}]
        stack=contextlib.AsyncExitStack()
        try:
            async with stack:
                mcp_client=None;native_tools=list(COMMON_TOOLS)
                if arm=='with-agentqakit':
                    server_env=child_env(workspace)
                    # This explicit package root prevents the public agentqa/
                    # spec folder from shadowing the editable Python package.
                    server_env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(workspace)])
                    stderr_path=str((folder/'mcp-stderr.log').resolve())
                    bootstrap=(f'import sys;sys.stderr=open({stderr_path!r},"w",encoding="utf-8");'
                               'from agentqa.cli import app;app()')
                    params=StdioServerParameters(command=str(PYTHON),args=['-B','-c',
                        bootstrap, 'mcp','serve','--workspace',str(workspace),
                        '--manifest','agentqa/agent-manifest.yaml'],env=server_env,cwd=workspace)
                    mcp_client=await stack.enter_async_context(Client(params,raise_exceptions=False,
                                                                     read_timeout_seconds=25))
                    manifest=resource_text(await mcp_client.read_resource('agentqa://project/manifest'))
                    native_tools+=await mcp_description(mcp_client)
                    messages.append({'role':'user','content':
                        'AgentQAKit is connected. The bounded manifest resource has already been read:\n'+manifest[:14000]+
                        '\nAvailable suite: agentqa/tests.yaml; target: agentqa/target.yaml; '
                        'gate: agentqa/quality-gate.yaml. All dependency writes are synthetic in-memory. '
                        'The user authorized their execution for this experiment: allow_side_effects=true, '
                        f'approval_id="competition-{case}". You may use run_tests and returned evidence '
                        'before and after edits. Internal AI generation/judge/analysis is disabled.'})
                dump(folder/'host-config.json',{'model':model,'arm':arm,'task_id':task_id,
                    'repeat':repeat,'tool_names':[t['function']['name'] for t in native_tools],
                    'limits':{'rounds':MAX_ROUNDS,'seconds':MAX_SECONDS,'tokens':MAX_TOKENS},
                    'temperature':0,'thinking':'disabled','sdk_auto_retries':0})
                for turn in range(MAX_ROUNDS):
                    elapsed=time.monotonic()-started
                    estimated=(sum(len(json.dumps(m,ensure_ascii=False)) for m in messages)+
                               len(json.dumps(native_tools,ensure_ascii=False)))//3
                    if elapsed>=MAX_SECONDS:status='time_budget_exhausted';break
                    if usage['total_tokens']+estimated+MAX_COMPLETION>MAX_TOKENS:
                        status='token_budget_exhausted';break
                    request_timeout=max(1,min(40,MAX_SECONDS-elapsed))
                    response=await client.chat.completions.create(model=model,messages=messages,
                        tools=native_tools,temperature=0,max_tokens=MAX_COMPLETION,
                        timeout=request_timeout,extra_body={'thinking':{'type':'disabled'}})
                    api_calls+=1
                    u=response.usage.model_dump() if response.usage else {}
                    for key in usage:usage[key]+=int(u.get(key) or 0)
                    provider.append({'id':response.id,'model':response.model,
                        'system_fingerprint':response.system_fingerprint,'usage':u,
                        'finish_reason':response.choices[0].finish_reason})
                    msg=response.choices[0].message
                    message={'role':'assistant','content':msg.content or ''}
                    calls=msg.tool_calls or []
                    if calls:message['tool_calls']=[c.model_dump(exclude_none=True) for c in calls]
                    messages.append(message)
                    log(timeline,{'event':'model','turn':turn+1,'elapsed_seconds':time.monotonic()-started,
                        'message':message,'usage':u,'response_id':response.id,'resolved_model':response.model})
                    for index,call in enumerate(calls):
                        name=call.function.name
                        try:
                            args=json.loads(call.function.arguments)
                            if index>=MAX_TOOL_CALLS_PER_ROUND:
                                result={'error':'Only five tool calls per turn are accepted'}
                            elif name=='finish':
                                finish=args;result={'submitted':True}
                            elif name in [t['function']['name'] for t in COMMON_TOOLS]:
                                result=await tools_host.call(name,args)
                            elif mcp_client is not None:
                                tools_host.usage['mcp_calls']+=1
                                result=await mcp_call(mcp_client,name,args)
                            else:result={'error':'Tool is unavailable in this arm'}
                        except Exception as exc:
                            # Local validation errors have safe text; transport errors do not.
                            result={'error':str(exc)[:350] if isinstance(exc,(ValueError,FileNotFoundError))
                                    else safe_exception(exc)}
                        text=json.dumps(result,ensure_ascii=False,default=str)
                        if len(text)>30000:text=text[:30000]+' [tool result truncated]'
                        messages.append({'role':'tool','tool_call_id':call.id,'content':text})
                        log(timeline,{'event':'tool','turn':turn+1,'name':name,
                            'arguments':call.function.arguments,'result':result,
                            'elapsed_seconds':time.monotonic()-started})
                    if finish is not None:status='completed';break
                    messages.append({'role':'user','content':
                        f'Budget update: {turn+1}/{MAX_ROUNDS} turns used, '
                        f'{usage["total_tokens"]}/{MAX_TOKENS} total tokens used. '
                        'If checks are sufficient, submit finish now; avoid redundant rereads.'})
                    if not calls:
                        messages.append({'role':'user','content':
                            'Continue with actual checks/edits, or use finish to submit your assessment.'})
                else:status='round_budget_exhausted'
        except BaseException as exc:
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
            status='runtime_error'
            log(timeline,{'event':'trial_error','error_type':safe_exception(exc)})
        elapsed=time.monotonic()-started
        if elapsed>MAX_SECONDS and status=='completed':status='time_budget_exhausted'
        if usage['total_tokens']>MAX_TOKENS and status=='completed':status='token_budget_exhausted'
        oracle=await evaluate_workspace(workspace)
        dump(folder/'after-oracle.json',oracle)
        edits={}
        for name,before in sources_before.items():
            after=(workspace/name).read_text(encoding='utf-8')
            if before!=after:
                import difflib
                edits[name]=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
                    fromfile='before/'+name,tofile='after/'+name))
        dump(folder/'patches.json',edits)
        record={'case_id':case,'task_id':task_id,'arm':arm,'repeat':repeat,'status':status,
            'initial_hash':info['initial_hash'],'initial_oracle_passed':pre.get('all_passed',False),
            'final_oracle_passed':oracle.get('all_passed',False),
            'failed_checks':oracle.get('failed_checks',[]),
            'high_risk_failures':oracle.get('high_risk_failures'),
            'release_ready':bool(finish.get('release_ready')) if finish else None,
            'defect_found_self_report':finish.get('defect_found') if finish else None,
            'finish':finish,'source_changed':bool(edits),'elapsed_seconds':round(elapsed,3),
            'budget_success':bool(oracle.get('all_passed') and elapsed<=MAX_SECONDS and usage['total_tokens']<=MAX_TOKENS),
            'api_calls':api_calls,'usage':usage,'tools':tools_host.usage,'provider_responses':provider}
        dump(folder/'result.json',record)
        log(output/'results.jsonl',record)
        print(json.dumps({'case':case,'status':status,'correct':record['budget_success'],
                         'seconds':record['elapsed_seconds'],'tokens':usage['total_tokens']},ensure_ascii=False),flush=True)
        return record

def summarize(output:Path,records:list[dict],protocol:dict)->dict:
    summary={'status':'ANALYZED','trial_count':len(records),'protocol_sha256':digest_file(output/'protocol.json'),
             'arms':{},'scope':'One synthetic task agent; bounded DeepSeek tool host, not a commercial IDE benchmark.'}
    for arm in ['without-agentqakit','with-agentqakit']:
        rows=[r for r in records if r['arm']==arm]
        defects=[r for r in rows if r['task_id']!='T06']
        healthy=[r for r in rows if r['task_id']=='T06']
        final_bad=[r for r in defects if not r['final_oracle_passed']]
        summary['arms'][arm]={
            'defect_trials':len(defects),'correct_repairs':sum(r['budget_success'] for r in defects),
            'correct_repair_rate':sum(r['budget_success'] for r in defects)/len(defects) if defects else None,
            'healthy_trials':len(healthy),'healthy_preserved':sum(r['budget_success'] for r in healthy),
            'healthy_false_alarms_self_report':sum(r['defect_found_self_report'] is True for r in healthy),
            'unsafe_final_versions':len(final_bad),
            'unsafe_release_recommendations':sum(r['release_ready'] is True for r in final_bad),
            'no_release_recommendation':sum(r['release_ready'] is None for r in rows),
            'mean_elapsed_seconds_all':sum(r['elapsed_seconds'] for r in rows)/len(rows) if rows else None,
            'total_tokens':sum(r['usage']['total_tokens'] for r in rows),
            'model_calls':sum(r['api_calls'] for r in rows),
            'budget_or_runtime_failures':sum(r['status']!='completed' for r in rows),
            'completed_sessions':sum(r['status']=='completed' for r in rows),
            'by_task':{t:{'runs':sum(r['task_id']==t for r in rows),
                'successes':sum(r['task_id']==t and r['budget_success'] for r in rows)} for t in TASK_IDS}}
    dump(output/'summary.json',summary)
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['case_id','task_id','arm','repeat','status','budget_success','final_oracle_passed',
                'release_ready','source_changed','elapsed_seconds','api_calls','total_tokens']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:writer.writerow({**{k:r.get(k) for k in fields},'total_tokens':r['usage']['total_tokens']})
    return summary

async def main_async(args):
    output=Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be new; no silent retry/resume')
    output.mkdir(parents=True,exist_ok=True)
    archive=output/'source';archive.mkdir()
    for name in ['repair_experiment.py','repair_fixture.py']:
        (archive/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    load_llm_environment(ROOT/'.env.local')
    model=os.environ['AGENTQA_LLM_MODEL']
    tasks=args.tasks.split(',') if args.tasks else list(TASK_IDS)
    schedule=[(t,a,r) for t in tasks for r in range(1,args.repeats+1)
              for a in ['without-agentqakit','with-agentqakit']]
    random.Random(20261117).shuffle(schedule)
    protocol={'version':'repair-paired-v1','phase':args.phase,'frozen_at_utc':
        __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'model':model,'temperature':0,'thinking':'disabled','trial_schedule':schedule,
        'budget':{'rounds':MAX_ROUNDS,'seconds':MAX_SECONDS,'tokens':MAX_TOKENS,
                  'per_call_output_tokens':MAX_COMPLETION,'sdk_auto_retries':0},
        'primary_metric':'Final code snapshot independently satisfies all checks within time/token budget; finish recommendation scored separately',
        'normal_control':'T06','script_sha256':digest_file(Path(__file__)),
        'fixture_sha256':digest_file(Path(__file__).with_name('repair_fixture.py')),
        'no_fixed_patch_substitution':True,'parallel_trials':args.concurrency,
        'failure_policy':'Retain all scheduled failures, no automatic retry.',
        'scoring':'Independent private oracle, unchanged public requirements and specifications.',
        'limits':'Pilot resource budget, no power or cross-project generalization claim.'}
    dump(output/'protocol.json',protocol)
    dump(output/'environment.json',{'python':sys.version,'python_executable':str(PYTHON),
        'platform':sys.platform,'model_alias':model,'endpoint_host':'api.deepseek.com',
        'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'env_secrets_saved':False})
    print(json.dumps({'event':'protocol_frozen','trials':len(schedule),'output':str(output)},ensure_ascii=False),flush=True)
    async with AsyncOpenAI(api_key=os.environ['AGENTQA_LLM_API_KEY'],
            base_url=os.environ['AGENTQA_LLM_BASE_URL'],max_retries=0,timeout=40) as client:
        sem=asyncio.Semaphore(args.concurrency)
        records=await asyncio.gather(*(trial(output,t,a,r,client,model,sem) for t,a,r in schedule))
    summary=summarize(output,records,protocol)
    print(json.dumps({'event':'completed','trials':len(records),'arms':summary['arms']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--tasks')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--phase',choices=['smoke','formal'],default='formal')
    asyncio.run(main_async(parser.parse_args()))
