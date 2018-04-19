## To produce the image by Deep learning Gan
##### 告訴系統要第幾張卡被看到。 Ex. 硬體總共有8張顯卡，以下設定只讓系統看到第1張顯卡
##### 若沒設定，則 Tensorflow 在運行時，預設會把所有卡都佔用
##### 要看裝置內顯卡數量及目前狀態的話，請在終端機內輸入 "nvidia-smi"
##### 若你的裝置只有一張顯卡可以使用，可以忽略此設定
###### os.environ["CUDA_VISIBLE_DEVICES"] = "0"
