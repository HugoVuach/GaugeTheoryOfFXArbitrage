"""Offline command line. No live trading interface is implemented."""
from pathlib import Path
import argparse
import json
from .data import ingest_csv, dataset_summary
from .pipeline import run_diagnostics, audit_fills
from .certification import conformal_quantile, minimum_blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--config", required=True)
    ingest.add_argument("--database", required=True)
    ingest.add_argument("files", nargs="+")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--database", required=True)
    detect = commands.add_parser("detect")
    detect.add_argument("--database", required=True)
    detect.add_argument("--config", required=True)
    detect.add_argument("--output", required=True)
    audit = commands.add_parser("audit-fills")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--currency", default="USD")
    replay = commands.add_parser("replay")
    replay.add_argument("--database", required=True)
    replay.add_argument("--alerts", required=True)
    replay.add_argument("--config", required=True)
    replay.add_argument("--output", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--scores", required=True, help="JSON list: one frozen-policy score per block")
    calibration.add_argument("--alpha", type=float, default=0.01)
    calibration.add_argument("--output", required=True)
    validation=commands.add_parser('validate-policies')
    validation.add_argument('--input',required=True)
    validation.add_argument('--registry',required=True)
    validation.add_argument('--output',required=True)
    validation.add_argument('--alpha',type=float,default=.05)
    validation.add_argument('--delta',type=float,default=.05)
    splits=commands.add_parser('split-plan')
    splits.add_argument('--database',required=True);splits.add_argument('--output',required=True)
    splits.add_argument('--training-days',type=int,default=60);splits.add_argument('--calibration-days',type=int,default=99)
    splits.add_argument('--test-days',type=int,default=30);splits.add_argument('--step-days',type=int,default=30)
    splits.add_argument('--purge-ms',type=int,default=5000)
    evaluation=commands.add_parser('evaluate-policies')
    evaluation.add_argument('--database',required=True);evaluation.add_argument('--alerts',required=True)
    evaluation.add_argument('--registry',required=True);evaluation.add_argument('--config',required=True)
    evaluation.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.command == "ingest":
        config = json.loads(Path(args.config).read_text())
        result = [ingest_csv(p, args.database, config) for p in args.files]
    elif args.command == "inspect":
        result = dataset_summary(args.database)
    elif args.command == "detect":
        result = run_diagnostics(args.database, json.loads(Path(args.config).read_text()), args.output)
    elif args.command == "audit-fills":
        result = audit_fills(args.input, args.output, args.currency)
    elif args.command == "replay":
        from .replay import replay_alerts
        result = replay_alerts(args.database, args.alerts, json.loads(Path(args.config).read_text()), args.output)
    elif args.command=='validate-policies':
        from .certification import validate_policy_blocks
        result=validate_policy_blocks(args.input,args.registry,args.output,args.alpha,args.delta)
    elif args.command=='split-plan':
        from .protocol import split_plan
        result=split_plan(args.database,args.output,args.training_days,args.calibration_days,args.test_days,args.step_days,args.purge_ms)
    elif args.command=='evaluate-policies':
        from .protocol import evaluate_hypothetical_policies
        result=evaluate_hypothetical_policies(args.database,args.alerts,args.registry,json.loads(Path(args.config).read_text()),args.output)
    else:
        q = conformal_quantile(json.loads(Path(args.scores).read_text()), args.alpha)
        result = {"alpha": args.alpha, "radius": q if q < float("inf") else None,
                  "radius_infinite": q == float("inf"), "minimum_blocks": minimum_blocks(args.alpha),
                  "assumption": "exchangeable block scores for a policy family frozen before calibration",
                  "not_claimed": "conditional coverage among accepted trades or validity under arbitrary drift"}
        Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
