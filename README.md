# PPNCKH
#Download file vgg16-hybrid1365_weights_notop.h5 từ link
Đặt hết vào một thư mục
Run: 
!python predict.py \
-image_path 'https://thoidai.com.vn/stores/news_dataimages/tra.nguyen/022020/04/17/2221_1.jpg' \
-model_path 'logistic_model_places.sav' \
-base_model 'places365'
