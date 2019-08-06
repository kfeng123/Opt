### ResNet

add_fpn_ResNet50_conv5_body

freezeat = 2

StopGradient is a helper operator that does no actual numerical computation, and in the gradient computation phase stops the gradient from being computed through it.


### How caffe2 works?

blobs: variables

workspace: contains blobs

create or switch to a workspace named "head":

>workspace.SwitchWorkspance("head", True)

Show the name list of the total blobs in the current workspace:

>workspace.Blobs()

fetch a blob to python:

>workspace.FetchBlob()


model: computational graph

model.net.Proto() describe the net structure

The content of a '.pbtxt' file is exactly the condtent of model.net.Proto()

Run the net once

>workspace.RunNet(model.net.Proto().name)

Once we run the net once, we can fetch and see all the blobs.

### ResNet 50

>workspace.FetchBlob('gpu_0/data').shape == (1, 3, 768, 1408)
>workspace.FetchBlob('gpu_0/retnet_cls_prob_fpn7').shape == (1, 45, 6, 11)

'gpu_0/pool1' is the output blob of the stem net

>workspace.FetchBlob('gpu_0/pool1').shape == (1, 64, 192, 352)

'res2' did NOT decrease the image size, but will increase the channel number. 'res2' contains 3 basic blocks. The output of 'res2' is 'gpu_0/res2_2_sum'

>workspace.FetchBlob('gpu_0/res2_2_sum').shape == (1, 256, 192, 352)

Note that there is a StopGradient op for 'gpu_0/res2_2_sum'. This implies that the blobs before 'gpu_0/res2_2_sum' will NOT be updated during training.

'res3' contins 4 basic blocks. The output of 'res3' is 'gpu_0/res3_3_sum'

>workspace.FetchBlob('gpu_0/res3_3_sum').shape == (1, 512, 96, 176)

'res4' contins 6 basic blocks. The output of 'res4' is 'gpu_0/res4_5_sum'

>workspace.FetchBlob('gpu_0/res4_5_sum').shape == (1, 1024, 48, 88)

'res4' contins 3 basic blocks. The output of 'res5' is 'gpu_0/res5_2_sum'

>workspace.FetchBlob('gpu_0/res5_2_sum').shape == (1, 2048, 24, 44)

### FPN

'gpu_0/fpn_6' is obtained from 'gpu_0/res5_2_sum' by a conv op with kernel 3 stride 2
'gpu_0/fpn_7' is obtained from 'gpu_0/fpn_6' by a Relu op and a conv op with kernel 3 stride 2

>workspace.FetchBlob('gpu_0/fpn_6').shape == (1, 256, 12, 22)
>workspace.FetchBlob('gpu_0/fpn_7').shape == (1, 256, 6, 11)

'gpu_0/res5_2_sum' with a conv with kernel 1 dim_out yields 'gpu_0/fpn_inner_res5_2_sum'

>workspace.FetchBlob('gpu_0/fpn_inner_res5_2_sum').shape == (1, 256, 24, 44)

'gpu_0/fpn_5' is obtained from 'gpu_0/fpn_inner_res5_2_sum' with a conv op with kernel 3

'gpu_0/fpn_4' ...

'gpu_0/fpn_3' ...

### cls_prob and bbox_pred

>workspace.FetchBlob('gpu_0/retnet_cls_prob_fpn3').shape == (1, 45, 96, 176) 

45 = 9 * 5 (5 is class number)

>workspace.FetchBlob('gpu_0/retnet_bbox_pred_fpn3').shape == (1, 36, 96, 176) 

36 = 9 * 4 (4 is (xmin, ymin, xmax, ymax))



![inference_0](https://github.com/kfeng123/Opt/tree/master/retinanet/fig/inference_0.png)
![inference_1](https://github.com/kfeng123/Opt/tree/master/retinanet/fig/inference_1.png)
![inference_2](https://github.com/kfeng123/Opt/tree/master/retinanet/fig/inference_2.png)
![inference_3](https://github.com/kfeng123/Opt/tree/master/retinanet/fig/inference_3.png)
![inference_4](https://github.com/kfeng123/Opt/tree/master/retinanet/fig/inference_4.png)
![inference_5](https://github.com/kfeng123/Opt/tree/master/retinanet/fig/inference_5.png)
![inference_6](https://github.com/kfeng123/Opt/tree/master/retinanet/fig/inference_6.png)
![inference_7](https://github.com/kfeng123/Opt/tree/master/retinanet/fig/inference_7.png)
![inference_8](https://github.com/kfeng123/Opt/tree/master/retinanet/fig/inference_8.png)
