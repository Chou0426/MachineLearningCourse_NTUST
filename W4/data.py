import os 
import shutil

path = 'dogs-vs-cats/test1/test1'
data = os.listdir(path)
cat_path = os.path.join(path,'cat')
dog_path = os.path.join(path,'dog')

for i,d in enumerate(data):
    org_path = os.path.join(path,d)

    if d =='cat' or d =='dog':
        pass
    else:
        label = d.split('.')[0]
        if label =='cat':
            new_path = os.path.join(cat_path,d)
        elif label =='dog':
            new_path = os.path.join(dog_path,d)
        shutil.move(org_path,new_path)

print('done')