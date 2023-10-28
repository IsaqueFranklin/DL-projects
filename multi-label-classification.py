from fastai.vision.all import *

path = untar_data(URLs.PASCAL_2007)
path.ls()

#Now we are using a CSV file to get the label for each image => using pandas library

df = pd.read_csv(path/'train.csv')
df.head()

dls = ImageDataLoaders.from_df(df, path, folder='train', valid_col='is_valid', label_delim=' ',
                               item_tfms=Resize(460), batch_tfms=aug_transforms(size=224))
dls.show_batch()

f1_macro = F1ScoreMulti(thresh=0.5, average='macro')
f1_macro.name = 'F1(macro)'
f1_samples = F1ScoreMulti(thresh=0.5, average='samples')
f1_samples.name = 'F1(samples)'
learn = vision_learner(dls, resnet50, metrics=[partial(accuracy_multi, thresh=0.5), f1_macro, f1_samples])

learn.lr_find()
learn.fine_tune(2, 3e-2)
learn.show_results()
learn.predict(path/'train/000005.jpg')