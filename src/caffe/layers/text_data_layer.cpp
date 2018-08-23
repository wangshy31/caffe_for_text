#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <map>

#include  <sys/mman.h>
#include  <sys/types.h>
#include  <sys/stat.h>
#include  <fcntl.h>

#include "stdint.h"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/text_data_layer.hpp"

#include "opencv2/opencv.hpp"


namespace caffe {
using std::min;
using std::max;



template <typename Dtype>
TextDataLayer<Dtype>::~TextDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void TextDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const TextDataParameter & param = this->layer_param_.text_data_param();
  if (param.has_crop_height() && param.has_crop_width()){
      crop_height_ = param.crop_height();
      crop_width_ = param.crop_width();
  }
  else{
      LOG(FATAL)<<"TextDataLayer: crop_height crop_width error!";
  }
  if (param.has_num_words()){
      num_words_ = param.num_words();
  }
  else{
      LOG(FATAL)<<"TextDataLayer: num_words_ error!";
  }
  if (param.has_flip())
    flip_ = param.flip();
  else
    flip_ = false;
  CHECK_EQ(crop_width_, 300)<<"crop_width_ should be equal to 300.";
  CHECK_GE(num_words_, crop_height_)<<"num_words should be greater or equal to crop_height_.";

  batch_size_ = param.batch_size();
  channel_ = param.channel();
  shuffle_ = param.shuffle();

  data_count = channel_ * crop_height_ * crop_width_;
  char buf[1024];

  // read label
  std::ifstream flabel(param.label_source().c_str());
  CHECK(flabel != NULL) << "file " <<param.label_source().c_str() << " read error" ;
  int num = 0;
  flabel.getline(buf,sizeof(buf));
  int len = sscanf(buf, "%d", &num);
  CHECK_GT(len, 0);
  text_info_.resize(num);
  int text_idx = 0;
  int temp_label = 0;
  while(flabel>>temp_label)
  {
    CHECK_LT(text_idx, num);
    text_info_[text_idx].label = temp_label;
    text_idx++;
  }
  flabel.close();
  CHECK_EQ(text_idx, num);
  LOG(INFO) <<"read label done ("<< num <<" samples)";

  // read list
  LOG(INFO) << "Reading content";
  std::ifstream fdata(param.data_source().c_str());
  char text_fn[10240];
  CHECK(fdata != NULL) << "fail to open " << param.data_source().c_str();
  text_idx = 0;
  while(!fdata.eof())
  {
    fdata.getline(text_fn, sizeof(text_fn));
    if (fdata.eof())
      break;
    CHECK_LT(text_idx, num);
    text_info_[text_idx].content = string(text_fn);
    text_idx++;
  }
  fdata.close();
  CHECK_EQ(text_idx, num);
  LOG(INFO) <<"read content done: #"<<text_info_.size()<<" samples.";

  // randomly shuffle data
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  if (shuffle_) {
	LOG(INFO) << "Shuffling data";
    ShuffleText();
  }

  current_row_ = 0;

  top[0]->Reshape(batch_size_, channel_, crop_height_, crop_width_);
  vector<int> label_shape(1, batch_size_);
  top[1]->Reshape(label_shape);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
	  << top[0]->channels() << "," << top[0]->height() << ","
	  << top[0]->width();
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(batch_size_, channel_, crop_height_, crop_width_);
    this->prefetch_[i].label_.Reshape(label_shape);
  }

  LOG(INFO)<<"Reading vector dict...";
  std::ifstream fdict(param.dict_source().c_str());
  CHECK(fdict != NULL) << "fail to open " << param.dict_source().c_str();
  Dtype word;
  while(fdict>>word)
    vec_dict.push_back(word);
  CHECK_EQ(vec_dict.size() % 300, 0)<<"vec_dict.size() % 300 should be equal to 0.";
  LOG(INFO)<<"vec_dict size: "<<vec_dict.size()/300.0;


}

template <typename Dtype>
void TextDataLayer<Dtype>::ShuffleText() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(text_info_.begin(), text_info_.end(), prefetch_rng);
}

template <typename Dtype>
void TextDataLayer<Dtype>::crop(string content, Dtype* data_out){
  Dtype *org_data = new Dtype[num_words_*crop_width_];
  if (content.empty())
    caffe_set(channel_*crop_height_*crop_width_, Dtype(0), data_out);
  else{
    std::istringstream istr;
    istr.str(content);
    int count = 0;
    int index = 0;
    while (istr>>index){
      if (count<num_words_){
        memcpy(org_data+count*crop_width_, &vec_dict[index*crop_width_], sizeof(Dtype) * crop_width_);
        count++;
      }
      else
        break;
    }
    for(int i=count;i<num_words_;i++)
    {
      memcpy(org_data+i*crop_width_, org_data+i%count*crop_width_, sizeof(Dtype) * crop_width_);
    }
  }
  if (this->phase_ == TRAIN) {
    caffe::rng_t* rng_seed =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    int h_off = (*rng_seed)() % (num_words_ - crop_height_);
    memcpy(data_out, org_data+h_off*crop_width_, sizeof(Dtype)*crop_height_*crop_width_);
  }
  else
      memcpy(data_out, org_data, sizeof(Dtype)*crop_height_*crop_width_);

  if (flip_){
    //Dtype *flip_data = new Dtype[crop_height_*crop_width_];
    for (int i = 0;i<crop_height_;i++)
      memcpy(data_out  + (i+crop_height_)* crop_width_, data_out+(crop_height_ - 1 - i)*crop_width_, sizeof(Dtype)*crop_width_);
      //memcpy(flip_data + i* crop_width_, data_out+(crop_height_ - 1 - i)*crop_width_, sizeof(Dtype)*crop_width_);
    //memcpy(data_out, flip_data, sizeof(Dtype)*crop_height_*crop_width_);
    //delete flip_data;
  }
  delete org_data;

}
// This function is called on prefetch thread
template <typename Dtype>
void TextDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  batch->data_.Reshape(batch_size_, channel_, crop_height_, crop_width_);
  vector<int> label_shape(1, batch_size_);
  batch->label_.Reshape(label_shape);
  Dtype *label_ptr_ = batch->label_.mutable_cpu_data();
  Dtype *data_ptr_ = batch->data_.mutable_cpu_data();
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
         crop(text_info_[current_row_].content, data_ptr_ + item_id*data_count);
         label_ptr_[item_id] = text_info_[current_row_].label;
         current_row_ ++;
         if(current_row_ >= text_info_.size()){
            current_row_ = 0;
	          if (shuffle_)
	     	    ShuffleText();
         }
  }
}



INSTANTIATE_CLASS(TextDataLayer);
REGISTER_LAYER_CLASS(TextData);

}  // namespace caffe
