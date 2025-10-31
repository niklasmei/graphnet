from graphnet.nm_corner.nm_collect_all import simple_gnn_encoder
from graphnet.nm_corner.nm_collect_all import general_compression_module
from graphnet.nm_corner.nm_collect_all import Theseus_DeepIce
from graphnet.nm_corner.nm_collect_all import DirectionRecoNM
from graphnet.nm_corner.nm_collect_all import OpeningAngleLoss

from graphnet.models import StandardModel
from typing import Dict, Any
from graphnet.data.dataset import SQLiteDataset
from graphnet.models.data_representation.graphs import KNNGraph
from graphnet.models.detector import IceCube86
from graphnet.data.utilities.sqlite_utilities import query_database
from graphnet.data import GraphNeTDataModule
from graphnet.training.labels import Direction


def pred_main(param_path, data_path=None, gpus=None
) -> None:
    
    #this is an example use that makes predictions with a model given via param_path
    #note that the Theseus_Deepice uses flash-attn and therefore absolutely needs a given gpu
    #also due to using flash-attn the flash-attn must be installed; follow guide on https://github.com/Dao-AILab/flash-attention


    features = ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'hlc']
    truths = ['energy', 'azimuth', 'zenith', 'position_x', 'position_y', 'position_z', 'event_no']

    config: Dict[str, Any] = {
        "num_workers": 15,
        "batch_size": 20,}
    

    graph_definition = KNNGraph(
    detector=IceCube86(),
    nb_nearest_neighbours=8,
    input_feature_names = features,
    )

    if data_path == None:
        nt_path = "/scratch/users/nikme/northern_tracks_data/dev_northern_tracks_muon_labels_v3/"
        db_path = nt_path+'dev_northern_tracks_muon_labels_v3_part_1.db'
    else:
        db_path = data_path
    sels = []
    query = "select event_no from truth limit 800"
    out = query_database(database=db_path, query=query)
    sels.append(out['event_no'].tolist())
    
    #defining the data
    dm = GraphNeTDataModule(
        dataset_reference=SQLiteDataset,
        dataset_args={
            "path": db_path,
            "pulsemaps" : "InIceDSTPulses",
            "truth_table" : 'truth',
            "features": features,
            "truth":truths,
            "data_representation" : graph_definition,
            "labels": {'direction': Direction()},  
        },
        selection=sels[0],
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "shuffle": True,
        },
        validation_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "shuffle": False,
        },
        test_selection=sels[0],
        test_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "shuffle": False,
        },
        train_val_split= [0.9, 0.1]
    )
    training_dataloader = dm.train_dataloader
    validation_dataloader = dm.val_dataloader
    test_dataloader = dm.test_dataloader

    #define general variables
    max_length=3840
    lat_dim=768

    #define encoding nets
    in_dim=5
    nb_messages=1
    lin_net_length=5
    ratio=0.5
  
    enc1 = simple_gnn_encoder(in_dim=in_dim,
                              lat_dim=lat_dim,
                              ratio=ratio,
                              nb_messages=nb_messages,
                              lin_net_length=lin_net_length)
    
    
    nb_nearest_enc = 12
    enc2 = simple_gnn_encoder(in_dim=in_dim,
                              lat_dim=lat_dim,
                              ratio=ratio,
                              nb_messages=nb_messages,
                              lin_net_length=lin_net_length,
                              nb_nearest=nb_nearest_enc)
    nb_nearest_enc = 20
    enc3 = simple_gnn_encoder(in_dim=in_dim,
                              lat_dim=lat_dim,
                              ratio=ratio,
                              nb_messages=nb_messages,
                              lin_net_length=lin_net_length,
                              nb_nearest=nb_nearest_enc)


    #actually subsampling - net_list is a list for subsampled encodings; empty means random subsample or based on hlc
    net_list=[enc1,enc2,enc3]
    nb_nearest=8
    scorer_length=5
    scorer_width=100
    hlc_pos=None
    random_subs_embedding=None
    
    
    
    compression = general_compression_module(max_length=max_length,
                                             lat_dim=lat_dim,
                                             nb_nearest=nb_nearest,
                                             net_list=net_list,
                                             scorer_length=scorer_length,
                                             scorer_width=scorer_width,
                                             hlc_pos=hlc_pos,
                                             random_subs_embedding=random_subs_embedding,) 
    
    backbone = Theseus_DeepIce(
        compression_model=compression,
        max_length=max_length,
        hidden_dim=lat_dim,
        exit_early=False) 
    latent_dim = backbone.nb_outputs

    task = DirectionRecoNM(
        hidden_size=latent_dim,
        target_labels=['direction'],
        loss_function=OpeningAngleLoss(),
    )


    model = StandardModel(data_representation = graph_definition,
                          backbone = backbone,
                          tasks = task,
                          )
    

    model.load_state_dict(param_path)

    print('selected gpu', gpus[0])
    xyz = ['x','y','z']
    cols = [f'direction_pred_{xyz[i]}' for i in range(len(xyz))]

    df = model.predict_as_dataframe(dataloader = test_dataloader,
                                    additional_attributes = truths,
                                    prediction_columns = cols,
                                    gpus = gpus)
    
    print(df.head(5))

if __name__ == "__main__":
    p_path = '/ptmp/mpp/nikme/best_w_snows_pretrain_plus_8p5mil_nt.pth'
    pred_main(param_path=p_path, data_path=None, gpus=[0])
