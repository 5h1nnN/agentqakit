"""Audit completed experiment artifacts and write descriptive, scoped results."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import os
import statistics
import sys
import importlib.metadata

from agentqa.comparison import load_run_artifact
from agentqa.evaluators.deepseek import load_llm_environment

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'benchmark/results/competition-20261117'
REPAIR=BASE/'repair'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def pct(x):return f'{x*100:.1f}%'

def main():
    protocol=read(REPAIR/'protocol.json')
    rows=[json.loads(l) for l in (REPAIR/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    expected={tuple(x) for x in protocol['trial_schedule']}
    actual={(r['task_id'],r['arm'],r['repeat']) for r in rows}
    assert expected==actual and len(rows)==len(actual)==36
    assert sha(REPAIR/'source/repair_experiment.py')==protocol['script_sha256']
    assert sha(REPAIR/'source/repair_fixture.py')==protocol['fixture_sha256']
    assert sha(Path(__file__).with_name('repair_fixture.py'))==protocol['fixture_sha256']
    initial_hashes=collections.defaultdict(set)
    native_runs=[]
    audited_tokens=collections.Counter()
    tool_errors=collections.Counter()
    for r in rows:
        folder=REPAIR/'trials'/r['case_id']
        assert read(folder/'result.json')==r
        initial_hashes[r['task_id']].add(r['initial_hash'])
        before=read(folder/'before-oracle.json');after=read(folder/'after-oracle.json')
        assert before['all_passed']==(r['task_id']=='T06')
        assert after['all_passed']==r['final_oracle_passed']
        assert len(after['checks'])==15
        assert sorted(after['failed_checks'])==sorted(r['failed_checks'])
        event_rows=[json.loads(l) for l in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        usage=collections.Counter()
        for e in event_rows:
            if e['event']=='model':
                for k in ['prompt_tokens','completion_tokens','total_tokens']:
                    usage[k]+=int(e['usage'].get(k) or 0)
            if e['event']=='tool' and isinstance(e.get('result'),dict) and 'error' in e['result']:
                tool_errors[r['arm']]+=1
        for k in ['prompt_tokens','completion_tokens','total_tokens']:
            assert usage[k]==r['usage'][k],(r['case_id'],k)
            audited_tokens[k]+=usage[k]
        for runs in (folder/'workspace').rglob('runs'):
            if not runs.is_dir():continue
            for run in runs.iterdir():
                if run.is_dir() and (run/'summary.json').exists():
                    artifact=load_run_artifact(run)
                    native_runs.append({'case_id':r['case_id'],'run_id':artifact.run_id,
                                        'path':str(run.relative_to(BASE))})
    assert all(len(h)==1 for h in initial_hashes.values())
    load_llm_environment(ROOT/'.env.local')
    secret=os.environ.get('AGENTQA_LLM_API_KEY','').encode()
    checked=0
    for folder in [REPAIR,BASE/'replay']:
        for p in folder.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if secret and secret in p.read_bytes():raise RuntimeError('Credential content found in experiment artifact')
            checked+=1
    hashes={str(p.relative_to(REPAIR)).replace('\\','/'):sha(p) for p in REPAIR.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name not in
        {'verification.json','artifact-checksums.json'}}
    dump(REPAIR/'artifact-checksums.json',hashes)
    verification={'status':'ANALYZED','scheduled_trials':36,'completed_records':len(rows),
        'same_initial_snapshot_across_arms_and_repeats':True,'independent_checks_per_final_snapshot':15,
        'model_usage_recomputed_from_call_receipts':dict(audited_tokens),
        'native_mcp_runs_loaded':len(native_runs),'native_runs':native_runs,
        'files_checksummed':len(hashes),'files_scanned_for_exact_credential':checked,
        'credential_findings':0,'tool_errors_retained':dict(tool_errors),
        'full_experiment_independently_rerun':False,
        'versions':{name:importlib.metadata.version(name) for name in ['agentqakit','mcp','openai','langgraph']}}
    dump(REPAIR/'verification.json',verification)
    summary=read(REPAIR/'summary.json')
    a=summary['arms']['without-agentqakit'];b=summary['arms']['with-agentqakit']
    ra=read(BASE/'replay/summary.json');rv=read(BASE/'replay/verification.json')
    assert ra['main_runs']==120 and rv['native_runs_loaded']==138
    difference=(b['correct_repair_rate']-a['correct_repair_rate'])*100
    def median(arm):return statistics.median(r['elapsed_seconds'] for r in rows if r['arm']==arm)
    task_names={'T01':'重试边界','T02':'明确确认','T03':'授权前读取','T04':'检索指令','T05':'幂等写入','T06':'正常控制'}
    task_table='\n'.join(f'| {t} {task_names[t]} | {a["by_task"][t]["successes"]}/{a["by_task"][t]["runs"]} | {b["by_task"][t]["successes"]}/{b["by_task"][t]["runs"]} |' for t in task_names)
    c=collections.Counter(r['status'] for r in rows)
    text=f'''# AgentQAKit 答辩实验结果（2026-11-17）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / descriptive collection
- Verification Status: ANALYZED
- Version Label: competition_results_v1
- 结果性质：实际执行与原始工件核验，不是预期数据。未独立完整重跑，不声称统计显著或跨项目泛化。

## 主要结果

两组试点均已执行。整体对照使用同一个受限 Coding Agent 工具主机与 DeepSeek Flash，通过真实模型调用修改合成校园项目；功能消融使用真实 AgentQAKit 执行与回放引擎。两者的评价对象和分母分开。

| 实验 | 指标 | 对照 | 使用项目或功能 | 差异 |
| --- | --- | --- | --- | --- |
| 整体使用 | 缺陷任务最终代码通过独立验收 | {a['correct_repairs']}/{a['defect_trials']}（{pct(a['correct_repair_rate'])}） | {b['correct_repairs']}/{b['defect_trials']}（{pct(b['correct_repair_rate'])}） | {difference:+.1f} 个百分点 |
| 严格回放 | 原始失败签名复现 | {ra['arms']['live']['reproduced']}/60（{pct(ra['arms']['live']['reproduction_rate'])}） | {ra['arms']['replay']['reproduced']}/60（{pct(ra['arms']['replay']['reproduction_rate'])}） | +58.3 个百分点 |
| 严格回放 | 上游依赖请求 | 60 次 | 0 次 | 少 60 次 |

第一行是固定预算结束时的代码质量，不等于所有会话都完成了发布建议。第二行是复现稳定性，不等于 Agent 安全率或错误发布减少。

## 实验一：整体使用对照

### 方法

- 目标：一个自包含、规则型的合成校园事务 Agent，提炼自 Campus Ops 的业务要求，不是 6 个生产 Agent。
- 任务：5 类单缺陷任务与 1 个正常控制，每组每任务 3 次，共 36 次模型会话。模型看不到标准修复版、缺陷标签和隐藏验收代码。
- 两组：同一 DeepSeek Flash 配置、同一源码快照和业务要求，均可读取代码、修改业务源码、编写/执行本地检查。不接入组不提供产品工具，接入组额外通过实际 stdio MCP 调用 AgentQAKit。
- 公开要求和 10 条公开测试规格两组均可读取。产品组可调用执行、结果、证据、比较和门禁；内部 AI 生成/Judge/分析未开启。
- 预算：每次最多 12 轮、240 秒、累计 60000 token，单次最多 4096 输出 token，SDK 自动重试关闭。调用前保守估算下一次请求并预留输出空间，因此可能在实际消耗小于 60000 时提前结束。规则两组相同。
- 验收：独立子进程使用私有的模拟服务、真实调用记录和写入账本执行 15 条检查，涵盖缺陷及正常业务。最终判定不读取平台自己的 PASS。
- 主指标：在时间/token预算内留下、并通过全部独立验收的最终代码快照。是否通过 finish 明确提交发布建议，另行统计。
- 顺序：预先冻结任务列表并以固定种子打乱调度。温度为0，但服务未提供可核验的逐次随机种子或固定模型版本ID，因此不宣称每次调用可完全复现。

### 分任务结果

| 任务 | 不接入 | 接入 AgentQAKit |
| --- | --- | --- |
{task_table}

分母为每类任务的3次运行。独立缺陷类型只有5类，重复运行不是新增的独立缺陷样本。

### 资源与交付完整性

| 指标（每组18次，含正常控制） | 不接入 | 接入 AgentQAKit |
| --- | ---: | ---: |
| 正常控制最终代码仍合格 | {a['healthy_preserved']}/3 | {b['healthy_preserved']}/3 |
| 明确提交 finish 的会话 | {a['completed_sessions']}/18 | {b['completed_sessions']}/18 |
| 未产生明确发布建议 | {a['no_release_recommendation']}/18 | {b['no_release_recommendation']}/18 |
| 模型调用次数 | {a['model_calls']} | {b['model_calls']} |
| 累计输入输出 token | {a['total_tokens']:,} | {b['total_tokens']:,} |
| 全部会话平均运行时间 | {a['mean_elapsed_seconds_all']:.2f} 秒 | {b['mean_elapsed_seconds_all']:.2f} 秒 |
| 全部会话中位运行时间 | {median('without-agentqakit'):.2f} 秒 | {median('with-agentqakit'):.2f} 秒 |

全体终止状态：{json.dumps(dict(c),ensure_ascii=False)}。预算终止和工具错误均保留，未删掉不成功的会话。最终代码已经合格但没有 finish 的会话可以计入代码质量主指标，不能宣称已完成发布流程。

本轮不足以稳定比较错误放行率：两组有大量会话尚未明确提交发布建议。不能将“没有建议”当作“正确拒绝”，也不能仅根据正常控制中的 defect_found 自报字段认定真实误报。

耗时包含模型、工具、MCP和检查流程，未把首次部署成本混入稳态单任务时间。两组以不同顺序交替并发运行，但无生产网络、真实团队或跨日期验证，不能据此泛化效率提升。未估算货币费用，token以模型服务返回的调用账单字段汇总。

### 可使用的结论

在本次5类受控缺陷、每类3次、相同预算的单项目试点中，接入组有 {b['correct_repairs']}/15 个最终代码快照通过独立验收，对照组为 {a['correct_repairs']}/15，观察差异为 {difference:+.1f} 个百分点。这个结果支持继续扩大样本，不能作为所有 Coding Agent 或所有真实项目的普遍提升幅度。

本轮模型宿主为实验中实现的受限工具循环，具备读文件、精确修改与本地检查能力，接入组实际调用 MCP。不能标注为 Codex、Claude Code 或 WorkBuddy 的产品对比。

## 实验二：严格回放消融

六个预先声明的可录制依赖场景：超时、缺字段、错值、过期记录、空检索和间接指令。使用一个小型确定性 Agent、真实 TestRunner/InteractionEngine/Cassette/schema evaluator 与 loopback HTTP 依赖。

普通重跑与严格回放保留相同代码、断言、包装、Trace和种子，只改变依赖来源。live计划预先包含全恢复、间歇和持续故障；不是把普通重跑全部设计为无法复现。

| 指标 | 普通重跑 | 严格回放 |
| --- | ---: | ---: |
| 主重执行 | 60 | 60 |
| 同一失败签名复现 | 25/60（41.7%） | 60/60（100%） |
| 上游 HTTP 请求 | 60 | 0 |
| 与独立业务 oracle 一致 | 60/60 | 60/60 |
| 回放 hit/miss | 不适用 | 60/0 |
| 预审固定补丁核验（另计） | 6/6 | 6/6 |

共120次主重执行，另6次原始录制、12次预审补丁核验，合计138次原生Runner运行。持续故障场景两组均10/10复现，符合“收益来自固定原始依赖条件”的解释。

普通重跑的35次未复现不能称为35次错误发布：依赖恢复后，当次行为确实符合当次输入。补丁核验两组均通过，本轮没有证明严格回放提高修复正确率或减少发布误判。

超时由依赖根据受控HTTP响应抛出TimeoutError，不代表真实TCP丢包。回放范围仅为已包装且已录制的依赖。这里没有模型调用，本机亚秒级耗时差异不作为生产提速或费用收益宣传。

## 试跑、异常与核验

- 正式修复对照前保留 repair-smoke 与 repair-smoke2。首次发现实验启动环境中的规范目录与可编辑安装包同名遮蔽，随后为MCP子进程明确设置包导入路径；单次输出上限从1800调到4096以允许完整工具JSON。两项均在正式协议冻结前完成，未修改生产核心实现。
- 主指标按既定“预算内独立验收的最终代码”执行，finish只用于另行统计发布建议，避免把已有正确代码仅因未提交总结而算成未修复。正式协议与源码存档说明该口径。
- 早期Windows受限沙箱对子进程/loopback存在启动限制；这些无正式数据的基础设施尝试保留说明，未混入正式样本。执行转到已有授权的本机进程环境；过期校准进程已核对并清理。
- LangChain OpenInference 在部分结构化输入上产生“无法解析消息类型”日志，保存在每次MCP的stderr文件中。没有静默删除平台或模型失败；独立验收不依赖这些日志的成功与否。
- 修复对照核验了36条原始记录、每个最终快照15条独立检查、输入快照一致性、模型token账单，以及 {len(native_runs)} 个原生MCP运行工件。
- 回放核验了138个原生运行与2122个冻结工件的SHA-256。两组结果均非完整独立复跑证明。
- 两组工件扫描未发现真实API密钥内容。模型只接收合成任务材料，候选代码与MCP子进程未继承模型密钥。

## 答辩呈现建议

可以分别呈现“相同预算下最终代码质量”和“相同失败条件的复现稳定性”。第一组同时标注单项目、5类缺陷、3次重复、受限Coding Agent，并展示未完成发布建议的数量；第二组标注6类合成依赖与每组60次重执行。

不使用“显著提升”“消除全部风险”“错误发布下降35次”等说法。不用Mutation Score替代上述独立代码验收和失败签名复现指标。正常版本的维护、成本与未完成项不能隐藏。

## 原始工件与复现入口

- 修复对照：`agentqakit/benchmark/results/competition-20261117/repair/`，含protocol、source存档、results.jsonl、summary、metrics、verification、artifact-checksums及逐次trials。
- 回放：`agentqakit/benchmark/results/competition-20261117/replay/`，含protocol、raw_results.jsonl、summary、metrics、verification、native、originals与repair_checks。
- 任务/验收校准：`agentqakit/benchmark/results/competition-20261117/repair-fixture-calibration/`。
- 脚本：`agentqakit/scripts/experiments/competition_20261117/`。正式修复脚本和fixture的准确版本另存于repair/source。
- 设计稿：`AgentQAKit_答辩实验重设计方案.md`。

重新执行需使用新的输出目录，不能覆盖本次数据：

```powershell
.venv\\Scripts\\python.exe -u scripts\\experiments\\competition_20261117\\repair_experiment.py --phase formal --repeats 3 --concurrency 2 --output benchmark/results/competition-20261117/repair-new
.venv\\Scripts\\python.exe -u scripts\\experiments\\competition_20261117\\replay_experiment.py --output benchmark/results/competition-20261117/replay-new
```

修复对照会调用现有模型API；上述命令用于说明可复现入口，报告生成不会重新执行模型实验。本轮未修改PPT或技术报告中的结果表。
'''
    report=ROOT.parent/'AgentQAKit_答辩实验结果_20261117.md'
    report.write_text(text,encoding='utf-8')
    print(json.dumps({'report':str(report),'repair_trials':len(rows),
        'repair_correct':{'without':a['correct_repairs'],'with':b['correct_repairs']},
        'native_mcp_runs_verified':len(native_runs),'files_checksummed':len(hashes),
        'exact_credential_findings':0},ensure_ascii=False))

if __name__=='__main__':main()
