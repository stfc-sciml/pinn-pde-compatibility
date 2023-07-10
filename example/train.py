import argparse
from pathlib import Path

from PDE import pde_obj_dict
from network import MembraneNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a MembraneNet')
    parser.add_argument('-o', '--OLHP', action='store_true',
                        help='turn on OLHP; default=false')
    parser.add_argument('-p', '--PDE', type=str, default='smooth',
                        help='target PDE; default=smooth')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='seed for model initialization; default=0')
    parser.add_argument('-b', '--beta_data', type=float, default=1.,
                        help='beta for data loss; default=1.0')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='device; default=cpu')

    args = parser.parse_args()
    if args.OLHP:
        res_path = Path(f'results/OLHP_{args.PDE}_seed{args.seed}')
        res_path.mkdir(parents=True, exist_ok=True)
        model = MembraneNet([8, 16, 32], OLHP=True, seed=args.seed)
        model.train_PDE(pde_obj_dict[args.PDE], n_side_train=128,
                        n_side_test=512, epochs=1000, lr=1e-4,
                        test_interval=5, save_dir=res_path, device=args.device,
                        beta_Omega=args.beta_data)
    else:
        res_path = Path(f'results/vanilla_{args.PDE}_seed{args.seed}')
        res_path.mkdir(parents=True, exist_ok=True)
        model = MembraneNet([8, 16, 32], OLHP=False, seed=args.seed)
        model.train_PDE(pde_obj_dict[args.PDE], n_side_train=128,
                        n_side_test=512, epochs=1000, lr=1e-4,
                        test_interval=5, save_dir=res_path, device=args.device,
                        beta_Omega=args.beta_data)
