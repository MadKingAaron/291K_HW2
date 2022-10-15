
s=1
k=5
w=32
p=0

def size(w, k, p, s):
    return ((w-k+2*p)/s)+1


s=2
k=2
w=10
p=0

print(size(w,k,p,s))