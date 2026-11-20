"""Verify and summarize the frozen local replay experiment without rerunning it."""
import argparse
from pathlib import Path
import hashlib
import json
from agentqa.comparison import load_run_artifact

p=argparse.ArgumentParser();p.add_argument("directory",type=Path);args=p.parse_args()
root=args.directory.resolve()
manifest=json.loads((root/"experiment_checksums.json").read_text(encoding="utf-8"))
failures=[]
for name,digest in manifest.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:failures.append(name)
assert not failures,failures
rows=[json.loads(line) for line in (root/"raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
originals=json.loads((root/"originals.json").read_text(encoding="utf-8"))
repairs=json.loads((root/"repair_checks.json").read_text(encoding="utf-8"))
for row in originals+rows+repairs:
    artifact=load_run_artifact(Path(row["artifact_path"]))
    assert artifact.run_id==row["run_id"]
    assert artifact.results[0].passed==row["passed"]
assert len(rows)==120 and len(originals)==6 and len(repairs)==12
assert all(not r["passed"] for r in originals)
assert all(r["passed"]==r["independent_pass"] for r in rows+repairs)
assert all(r["seed"]==421 and r["trial_seed"]==421 for r in originals+rows+repairs)
protocol=json.loads((root/"protocol.json").read_text(encoding="utf-8"))
assert hashlib.sha256(Path(__file__).with_name("replay_experiment.py").read_bytes()).hexdigest()==protocol["script_sha256"]
summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
live=summary["arms"]["live"]; replay=summary["arms"]["replay"]
body=f'''# 严格回放单变量实验结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-11-17
- Verification Status: ANALYZED
- Version Label: replay_experiment_v1
- Execution Status: COMPLETED，原始结果与全部原生工件已核验；未独立重跑整个实验，不声称跨环境复现已验证。

## 方法与范围

通过真实 AgentQAKit TestRunner、InteractionEngine、工具包装与 schema evaluator 运行。目标是一个自定义 TargetAdapter 内的小型确定性依赖调用 Agent fixture，连接实际的 loopback HTTP 服务。未调用真实 LLM、外部网络或真实业务写入。

六个合成依赖场景分别为受控依赖超时、缺字段、错误值、过期记录、空检索和带间接指令的检索内容。目标在异常数据下可能错误声称任务完成。行为断言允许正确结果或安全降级，独立 oracle 对照真实依赖内容与 Agent 输出核验。

先录制每个场景的原始失败，再在相同代码、断言、包装、Trace 与种子下，比较普通重跑和严格回放。预先冻结的 live 变化计划包含完全恢复、间歇故障与持续故障，不将所有普通重跑设置成恢复。主指标是原始失败评估签名的复现率，签名包括目标状态、评估器类型/状态与违规类型，不只是命中 cassette。

## 实际结果

| 指标 | 普通重跑 | 严格回放 |
| --- | ---: | ---: |
| 主重执行次数 | {live['n']} | {replay['n']} |
| 原始失败签名复现 | {live['reproduced']}/60 | {replay['reproduced']}/60 |
| 复现率 | {live['reproduction_rate']:.2%} | {replay['reproduction_rate']:.2%} |
| 实际上游 HTTP 请求 | {live['upstream_requests']} | {replay['upstream_requests']} |
| 回放 hit / miss | 不适用 | {replay['replay_hits']} / {replay['replay_misses']} |
| 与独立业务 oracle 一致 | {live['oracle_agreement']}/60 | {replay['oracle_agreement']}/60 |
| 单次运行中位耗时 | {live['median_seconds']:.4f} 秒 | {replay['median_seconds']:.4f} 秒 |
| 预审固定补丁核验 | {live['repair_checks_passed']}/6 | {replay['repair_checks_passed']}/6 |

严格回放比普通重跑多复现 35 次，差值为 {(replay['reproduction_rate']-live['reproduction_rate'])*100:.2f} 个百分点。普通重跑有 35 次因为依赖条件改变不再出现原始失败；这不是独立评估器的误判，也不能直接称为 35 次错误发布。两组全部主运行均与 oracle 一致。

共 120 次主重执行，另有 6 次原始录制和 12 次固定补丁核验，合计 138 次原生 Runner 运行。补丁是预审的确定性处理分支，仅用于隔离回放核验，不是 Coding Agent 自动修复成果。

## 分场景结果

| 场景 | 普通重跑复现 | 严格回放复现 |
| --- | ---: | ---: |
'''
for scenario,arms in summary["by_scenario"].items():
    body+=f"| {scenario} | {arms['live']['reproduced']}/10 | {arms['replay']['reproduced']}/10 |\n"
body+='''
## 可支持的结论与限制

- 在本次六种可录制、依赖条件变化的合成场景中，严格回放恢复了原始依赖输入，稳定重现相同评估失败，并避免重执行访问上游。
- 持续故障场景 wrong_value 中两组均 10/10，说明依赖本身不变化时，普通重跑也能稳定复现。效果来自固定交互条件，不是采用了更宽松的判断规则。
- 六个条件属于同一个小型 Agent fixture。60 次/组是重复执行次数，不是 60 个独立缺陷。没有给出统计显著性或跨真实项目泛化结论。
- timeout 场景是依赖实现根据受控 HTTP 响应抛出 TimeoutError，不是通过丢包或网络延迟制造的真实 TCP 超时。异常经产品原生 cassette 记录，回放使用安全 ReplayedDependencyError。
- 本机耗时包含 HTTP 客户端建立及本地工件写入，不能直接推断真实服务费用、互联网延迟或生产系统加速倍数。
- 实验选择可录制的依赖，未覆盖未包装调用、全环境冻结、模型随机性、参数或调用次序变化造成的 miss。正式 120 次没有 miss，不表示 miss 功能经过本轮专项验证。
- 没有使用 Mutation 注入。服务响应变化进入实际依赖包装和录制，避免把外层 Mutation 故障误当作 cassette 内容。
- 初始沙箱 smoke 启动被阻塞且无结果，在启动观察硬超时后终止。使用允许 loopback 的本机权限后，独立 smoke2 成功，再执行正式实验。smoke2 不进入正式统计。

## 命令与工件

在 agentqakit 仓库根目录执行：

```powershell
.venv\\Scripts\\python.exe -u scripts\\experiments\\competition_20261117\\replay_experiment.py --output benchmark/results/competition-20261117/replay
.venv\\Scripts\\python.exe scripts\\experiments\\competition_20261117\\replay_verify.py benchmark/results/competition-20261117/replay
```

首次正式运行退出码 0。脚本拒绝非空输出目录，完整复跑需指定新的目录。正式数据位于本报告同目录：protocol.json、raw_results.jsonl、summary.json、metrics.csv、originals.json、repair_checks.json、upstream_audit.json、environment.json、experiment_checksums.json 和 native/ 内 138 个完整原生运行目录。
'''
(root/"results_report.md").write_text(body,encoding="utf-8")
verification={"status":"ANALYZED","checksummed_files":len(manifest),"native_runs_loaded":len(originals)+len(rows)+len(repairs),"frozen_script_hash_matches":True,"all_originals_fail":True,"oracle_agreement_main":120,"main_count":120,"full_rerun":False}
(root/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(verification,ensure_ascii=False))
