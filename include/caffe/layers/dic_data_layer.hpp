#ifndef CAFFE_DIC_DATA_LAYER_HPP_
#define CAFFE_DIC_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from text files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class DicDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DicDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}

  virtual ~DicDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "DicData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }
 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleText();
  virtual void load_batch(Batch<Dtype>* batch);

  void crop(string, Dtype *, int crop_height_, int crop_width_);
  struct textinfo{
    int label;
    string content;
  };
  vector<textinfo> text_info_;
  int current_row_;
  int data_count;

  size_t batch_size_;
  int channel_, crop_height_, crop_width_;
  bool shuffle_;

  vector<Dtype> vec_dic;
};
}  // namespace caffe

#endif  // CAFFE_DIC_DATA_LAYER_HPP_
