import os

L1=set(os.listdir('2nd data'))
L2=set(map(lambda x : x.split()[-1],os.listdir('Contrast FFR_BMP')))
L3=set(map(lambda x : x.split()[-1],os.listdir('8명 추가 자료')))
L4=set(map(lambda x : x.split()[-1],os.listdir('14명 환자 압축 영상')))

print(len(L1),len(L2),len(L3),len(L4))

T=L1,L2,L3,L4
for i in range(len(T)):
    for j in range(i+1,len(T)):
        print(i+1,j+1,len(T[i]&T[j]))
        print(T[i]&T[j])
print(L1&L2 == L1,L1&L2 == L2)
