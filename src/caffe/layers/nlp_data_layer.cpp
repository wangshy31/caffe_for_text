#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include  <sys/mman.h>
#include  <sys/types.h>
#include  <sys/stat.h>
#include  <fcntl.h>

#include "stdint.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
//#include "caffe/data_layers.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/nlp_data_layer.hpp"

#include "opencv2/opencv.hpp"


namespace caffe {
using std::min;
using std::max;



template <typename Dtype>
NlpDataLayer<Dtype>::~NlpDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void NlpDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_t  channel_[2], height_[2], width_[2];
  const NlpDataParameter & param = this->layer_param_.nlp_data_param();

  if (param.has_crop_size() && (!param.has_crop_height()) && (!param.has_crop_width())){
    crop_height_ = param.crop_size();
    crop_width_ = param.crop_size();
  }
  else{
    if ((!param.has_crop_size()) && param.has_crop_height() && param.has_crop_width()){
      crop_height_ = param.crop_height();
      crop_width_ = param.crop_width();
    }
    else{
      LOG(FATAL)<<"NlpDataLayer: crop_size crop_height crop_width error!";
    }
  }

  batch_size_ = param.batch_size();
  channel_[0] = param.channel();
  height_[0] = crop_height_;
  width_[0] = crop_width_;
  shuffle_ = param.shuffle();

  channel_[1] = 1;
  height_[1] = 1;
  width_[1] = 1;
  data_count = height_[0]*width_[0]*channel_[0];
  char buf[1024];

  // read label
  std::ifstream flabel(param.label_source().c_str());
  CHECK(flabel != NULL) << "file " <<param.label_source().c_str() << " read error" ;
  int num = 0;
  int num_people = 0;
  flabel.getline(buf,sizeof(buf));
  int len = sscanf(buf, "%d%d", &num, &num_people);
  CHECK_GT(len, 0);
  nlp_info_.resize(num);
  int image_idx = 0;
  int temp_label = 0;
  while(flabel>>temp_label)
  {
    CHECK_LT(image_idx, num);
    nlp_info_[image_idx].label = temp_label;
    image_idx++;
  }
  flabel.close();
  CHECK_EQ(image_idx, num);
  if(len == 1){
    LOG(WARNING) << "Using deprecated label.meta format. Not specifying #people is unsafe";
    num_people = temp_label+1;
  }
  LOG(INFO) <<"read label done ("<< num_people <<" people)";

    // read list
  LOG(INFO) << "Reading list";
  string image_prefix = param.image_prefix().c_str();
  std::ifstream fdata(param.data_source().c_str());
  char image_fn[1024];
  CHECK(fdata != NULL) << "fail to open " << param.data_source().c_str();
  image_idx = 0;
  while(fdata>>image_fn)
  {
    CHECK_LT(image_idx, num);
    nlp_info_[image_idx].filename = image_prefix + "/" + string(image_fn);
    image_idx++;
  }
  fdata.close();
  CHECK_EQ(image_idx, num);
  LOG(INFO) <<"read list done (#image, #people: "<<nlp_info_.size()<<", "<<num_people<<")";

  // randomly shuffle data
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  if (shuffle_) {
	LOG(INFO) << "Shuffling data";
	ShuffleImages();
  }

  //const int thread_id = Caffe::getThreadId();
  //const int thread_num = Caffe::getThreadNum();
  //current_row_ = nlp_info_.size() / thread_num * thread_id;
  current_row_ = 0;

  // getchar();
  // Reshape blobs.
  top[0]->Reshape(batch_size_, channel_[0], height_[0], width_[0]);
  top[1]->Reshape(batch_size_, channel_[1], height_[1], width_[1]);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
	  << top[0]->channels() << "," << top[0]->height() << ","
	  << top[0]->width();
  //this->prefetch_data_.Reshape(batch_size_ / thread_num, channel_[0], height_[0], width_[0]);
  //this->prefetch_label_.Reshape(batch_size_ / thread_num, channel_[1], height_[1], width_[1]);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(batch_size_, channel_[0], height_[0], width_[0]);
    this->prefetch_[i].label_.Reshape(batch_size_, channel_[1], height_[1], width_[1]);
  }

}

template <typename Dtype>
void NlpDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(nlp_info_.begin(), nlp_info_.end(), prefetch_rng);
}

template <typename Dtype>
void NlpDataLayer<Dtype>::crop(string path,Dtype* data_out){
  cv::Mat img(crop_height_,crop_width_,CV_32FC1,1);
  std::ifstream fin(path.c_str());
  //LOG(INFO)<<path;
  if(!fin){
    LOG(FATAL) <<path<<" data failed to read!";
  }
  int count = 0;
  int wordcount = 0;
  for (int i  = 0;i<crop_height_;i++)
  {
    for(int j = 0;j<crop_width_;j++)
    {
      float tmp;
      if(fin>>tmp)
      {
        data_out[count] = tmp;
        wordcount++;
      }
      else
      {
        CHECK_GT(count,0)<<"count must be greater than 0"<<path;
        data_out[count] = data_out[count%wordcount];
      }
      count++;

      //LOG(INFO)<<img.at<float>(i,j);
    }
  }
  fin.close();
 }
// This function is called on prefetch thread
template <typename Dtype>
void NlpDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  // datum scales
  //const int lines_size = lines_.size();
  batch->data_.Reshape(batch_size_, 1, crop_height_, crop_width_);
  batch->label_.Reshape(batch_size_, 1,1,1);
  Dtype *label_ptr_ = batch->label_.mutable_cpu_data();
  Dtype *data_ptr_ = batch->data_.mutable_cpu_data();
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    // get a blob
       //for (int i = 0; i < batch_size_ / Caffe::getThreadNum(); ++i){
         crop(nlp_info_[current_row_].filename, data_ptr_ + item_id*data_count);
         label_ptr_[item_id] = nlp_info_[current_row_].label;
         //LOG(INFO)<<nlp_info_[current_row_].filename<<" "<<nlp_info_[current_row_].label;
         current_row_ ++;
         if(current_row_ >= nlp_info_.size()){
           current_row_ = 0;
	       if (shuffle_)
	     	  ShuffleImages();
         }
       //}
  }
}



INSTANTIATE_CLASS(NlpDataLayer);
REGISTER_LAYER_CLASS(NlpData);

}  // namespace caffe
