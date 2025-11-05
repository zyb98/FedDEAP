import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='pacs',help='domainnet or office_caltech10')
    parser.add_argument('--data_path', default='',help='path of each dataset')
    parser.add_argument('--seed', default=123,type=int)
    parser.add_argument('--round',  default=100, type=int,help='repeat times')
    parser.add_argument('--batch_size', default=64,type=int)
    parser.add_argument('--lr', default=0.001,type=float,help='learing rate')
    parser.add_argument('--gctx', default=16,type=int,help='the length of prompts')
    parser.add_argument('--lamda', default=1, type=float)
    parser.add_argument('--sigma', default=1, type=float)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Choose device: 'cuda:0', 'cuda:1', ..., or 'cpu' ")
    return parser
