from .videoxflow import VideoXFlow
from .videoyflow import VideoYFlow
from .videoyflows import VideoYFlowS
from .videoyflowr import VideoYFlowR
# from .videoydcn import VideoYDCN
from .videoyflowbase import VideoYFlowBase
from .videoybase2 import VideoYBase2

def getModel(args, is_eval=False):
    if args.model == 'xflow':
        return (VideoXFlow(channel=args.N, e_num=args.e_num, p_num=args.p_num, is_eval=is_eval), args.e_num + args.p_num)
    if args.model == 'yflow':
        return (VideoYFlow(channel=args.N, e_num=args.e_num, p_num=args.p_num, is_eval=is_eval), args.e_num + args.p_num)
    if args.model == 'ybase':
        return (VideoYBase2(channel=args.N, e_num=args.e_num, p_num=args.p_num, is_eval=is_eval), args.e_num + args.p_num)
    if args.model == 'yflows':
        return (VideoYFlowS(channel=args.N, e_num=args.e_num, p_num=args.p_num, is_eval=is_eval), args.e_num + args.p_num)
    if args.model == 'yflowr':
        return (VideoYFlowR(channel=args.N, e_num=args.e_num, p_num=args.p_num, is_eval=is_eval), args.e_num + args.p_num)
    if args.model == 'yflowb':
        return (VideoYFlowBase(channel=args.N, e_num=args.e_num, p_num=args.p_num, is_eval=is_eval), args.e_num + args.p_num)
    # if args.model == 'ydcn':
        # return (VideoYDCN(channel=args.N, e_num=args.e_num, p_num=args.p_num, is_eval=is_eval), args.e_num + args.p_num)
