import torch

def get_model(args):
    model_name = args['model_name']
    pretrained = args.get('pretrained', False)
    num_slices = args.get('num_slices', None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_name == 'UNet':
        from monai.networks.nets import UNet
        from monai.networks.layers import Norm, Act
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            act=Act.RELU,
            dropout=args.get('dropout', 0.2)
        ).to(device)

    elif model_name == 'DynUNet':
        from monai.networks.nets import DynUNet
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=(3, 3, 3, 3),
            strides=(1, 2, 2, 2),
            upsample_kernel_size=(2, 2, 2),
            res_block=True,
            dropout=args.get('dropout', 0.1)
        ).to(device)

    elif model_name == 'GroupNormUNet':
        from monai.networks.nets import UNet
        from monai.networks.layers import Norm, Act
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=("group", {"num_groups": args.get('num_groups', 2)}),
            act=Act.RELU,
            dropout=args.get('dropout', 0.2)
        ).to(device)

    elif model_name == 'AHNet':
        from monai.networks.nets import AHNet
        model = AHNet(
            layers=args.get('layers', (3,4,6,3)),
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            psp_block_num=args.get('psp_block_num',4),
            upsample_mode='transpose',
            pretrained=args.get('pretrained', True)
        ).to(device)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model
