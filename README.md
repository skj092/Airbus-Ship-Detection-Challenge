

Resnet34 is commonly used as an encoder for U-net and SSD, boosting the model performance and training time since you do not need to train the model from scratch.
However, in particular cases it makes sense to do fine-tuning of Resnet34 model before using it as a decoder for object localization or image segmentation.
In this competition the size of ship masks is much smaller than the size of images that leads to quite unbalanced training with ~1 positive pixel per 1000 negative ones.
If images with no ships are used, instead of ~1:1000 you will end up with ~1:10000 unbalance, which is quite tough.
Moreover, the training time is ~4 times longer since you need to process more images in each epoch.
So, it is reasonable to drop empty images and focus only on ones with ships.

Meanwhile, since the current dataset is quite different from ImageNet, the empty images are quite helpful in fine-tuning your encoder on a pseudo task - ship detection.
Moreover, when the training of your U-net or SSD model is completed, you can run the model on images without ships,
add false positives (~4000 in my case) as negative example to you training set, and train the model for several additional epochs.
Finally, a good model focused on a single task, ship detection, can boost the final score when you stack up it with U-net or SSD.
If you predict a ship for an empty image you will get automatically zero score for it, and since PLB has ~85% of empty images, prediction of empty images is quite important.

In this notebook I want to share how to pretrain Resnet34 (or higher end models) on a ship detection task.
After training of the head layers of the model on 256x256 rescaled images for one epoch the accuracy has reached 93.7%.
The following fine-tuning of entire model for 2 more epochs with learning rate annealing boosted the accuracy to ~97%.
If the training is continued for several epochs with a new data set composed of images of 384x384 resolution, the accuracy could be boosted to ~98%.
Unfortunately, continuing training the model on full resolution, 768x768, images leaded to reduction of the accuracy that is likely attributed to insufficient model capacity.

