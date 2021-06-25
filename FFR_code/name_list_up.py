
import os

def search(dirname):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == target: 
                    L.append(full_filename.split('\\')[-1])
                    
    except PermissionError:
        pass

def search_count(dirname):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search_count(full_filename)
                
            else:
                for i in range(len(names)):
                    if names[i] == full_filename.split('\\')[-1]:
                        nums[i]+=1
    except PermissionError:
        pass

target = ".bmp"

L=[]
#search("14명 환자 압축 영상")
search("2nd data")

names=list(set(L))
#print(names)
nums=[0]*len(names)

#search_count("14명 환자 압축 영상")
search_count("2nd data")

print(dict(zip(names,nums)))
print(f'종류 : {len(names)}개, 총 개수 : {sum(nums)}')
