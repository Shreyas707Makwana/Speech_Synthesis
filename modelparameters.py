model_filename = 'test'
Training_file = "filelists/list.txt" 
hparams.training_files = Training_file
hparams.validation_files = Training_file



hparams.p_attention_dropout=0.1
hparams.p_decoder_dropout=0.1
hparams.decay_start = 15000         
hparams.A_ = 3e-4 
hparams.B_ = 8000                   
hparams.C_ = 0                      
hparams.min_learning_rate = 1e-5    

generate_mels = True
hparams.show_alignments = True
alignment_graph_height = 600
alignment_graph_width = 1000


hparams.batch_size =  6
hparams.load_mel_from_disk = True
hparams.ignore_layers = [] 

hparams.epochs =  250

torch.backends.cudnn.enabled = hparams.cudnn_enabled
torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

output_directory = '/content/drive/MyDrive/colab/outdir' 
log_directory = '/content/TTS-TT2/logs' 
log_directory2 = '/content/drive/My Drive/colab/logs' 
checkpoint_path = output_directory+(r'/')+model_filename
hparams.text_cleaners=["english_cleaners"] + (["cmudict_cleaners"] if use_cmudict is True else [])
