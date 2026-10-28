import argparse
import json
from . import pipeline,model,data

def main():
    parser=argparse.ArgumentParser(description='MaintAI local operations')
    sub=parser.add_subparsers(dest='command',required=True)
    init=sub.add_parser('bootstrap'); init.add_argument('--csv')
    sub.add_parser('worker'); sub.add_parser('tick'); sub.add_parser('status'); sub.add_parser('rollback')
    replay=sub.add_parser('replay'); replay.add_argument('--count',type=int,default=250); replay.add_argument('--scenario',choices=['normal','high-wear'],default='normal')
    sub.add_parser('evaluate-test',help='Evaluate the current model on the final test split; do not use this result to tune it')
    args=parser.parse_args()
    if args.command=='bootstrap': result=pipeline.bootstrap(args.csv)
    elif args.command=='worker': return pipeline.worker()
    elif args.command=='tick': result=pipeline.tick()
    elif args.command=='status': result=pipeline.status()
    elif args.command=='rollback': result=pipeline.rollback()
    elif args.command=='replay':
        if not 1<=args.count<=500: parser.error('--count must be between 1 and 500')
        result=pipeline.replay(args.count,args.scenario)
    else:
        data.verify_splits()
        result={'split':'untouched_test','metrics':model.evaluate(model.load(),data.read_split('test'))}
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
