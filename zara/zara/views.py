from django.shortcuts import render

def indice(request):
    return render(request, 'index.html')


def aceptar(request):
    return render(request, 'resultados.html')