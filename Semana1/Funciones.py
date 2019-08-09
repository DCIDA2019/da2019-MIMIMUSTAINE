def f2(x,y):
    def cuadrado(a):
        return a * a
    tmp = cuadrado(x) + cuadrado(y)
    return x,y,tmp
