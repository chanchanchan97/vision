#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

template <typename scalar_t>
at::Tensor nms_kernel_impl(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  TORCH_CHECK(!dets.is_cuda(), "dets must be a CPU tensor");
  TORCH_CHECK(!scores.is_cuda(), "scores must be a CPU tensor");
  TORCH_CHECK(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong));

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));  //将置信度分数按照从大到小的顺序排列，返回scores中相应的下标索引

  auto ndets = dets.size(0);  //预测框数量
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));  //初始化
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));  //初始化

  auto suppressed = suppressed_t.data_ptr<uint8_t>();  //被舍弃的预测框
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();  //置信度分数从大到小排列的列表索引
  auto x1 = x1_t.data_ptr<scalar_t>();  //左上角横坐标
  auto y1 = y1_t.data_ptr<scalar_t>();  //左上角纵坐标
  auto x2 = x2_t.data_ptr<scalar_t>();  //右下角横坐标
  auto y2 = y2_t.data_ptr<scalar_t>();  //右下角纵坐标
  auto areas = areas_t.data_ptr<scalar_t>();  //预测框面积

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {  //按照置信度分数从大到小的顺序，遍历所有预测框
    auto i = order[_i];  //按照置信度分数从大到小的顺序，获取第_i个scores的下标
    if (suppressed[i] == 1)  //如果这个预测框已经被舍弃，则进行下一个预测框
      continue;
    keep[num_to_keep++] = i;
    auto ix1 = x1[i];  //该预测框左上角横坐标
    auto iy1 = y1[i];  //该预测框左上角纵坐标
    auto ix2 = x2[i];  //该预测框右下角横坐标
    auto iy2 = y2[i];  //该预测框右下角纵坐标
    auto iarea = areas[i];  //该预测框面积

    for (int64_t _j = _i + 1; _j < ndets; _j++) {  //将该预测框与剩余预测框对比
      auto j = order[_j];
      if (suppressed[j] == 1)  //如果这个预测框已经被舍弃，则进行下一个预测框
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
      auto inter = w * h;  // 两个预测框的交集面积
      auto ovr = inter / (iarea + areas[j] - inter);  // iou交并比
      if (ovr > iou_threshold)  //如果iou大于阈值，则将预测框舍弃
        suppressed[j] = 1;
    }
  }
  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

at::Tensor nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  auto result = at::empty({0}, dets.options());

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_kernel", [&] {
    result = nms_kernel_impl<scalar_t>(dets, scores, iou_threshold);
  });
  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
}

} // namespace ops
} // namespace vision
