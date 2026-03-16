dataset_paths = {
    'flowers_train': '/root/c_1206/CAE/filelists/102flowers/base.json',
    'flowers_test': '/root/c_1206/CAE/filelists/102flowers/val.json',
    'vgg_faces_train': '/root/c_1206/CAE/filelists/vggfaces_news/base.json',
    'vgg_faces_test': '/root/c_1206/CAE/filelists/vggfaces_news/val.json',
    'animal_faces_train': '/root/c_1206/CAE/filelists/animals/base.json',
    'animal_faces_test': '/root/c_1206/CAE/filelists/animals/val.json',
}

model_paths = {
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
    'stylegan_flowers': 'pretrained_models/stylegan2-flowers.pt',
    'stylegan_animalfaces': 'pretrained_models/stylegan2-animalfaces.pt',
    'stylegan_vggfaces': 'pretrained_models/psp_vggfaces.pt',
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
    'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
    'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
    'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar',
}
