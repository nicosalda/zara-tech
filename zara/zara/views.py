from django.shortcuts import render, redirect
from zara.photo import UploadFileForm, handle_photo

# Index
def indice(request):
    # Creamos el formulario
    form = UploadFileForm()
    # Renderizamos el template e insertamos el formulario
    return render(request, 'index.html', {'form': form})

# Metodo para recibir la foto, procesarla y devolver las 
# imagenes similares.
def foto(request):
    # Comprobamos que el metodo que recibimos es de tipo POST
    if request.method == "POST":
        # Parseamos el formulario
        form = UploadFileForm(request.POST, request.FILES)
        # Comprobamos que el formulario es valido
        if form.is_valid():
            # Buscamos las imagenes similares
            type_of_clothing = handle_photo(request.FILES['file'])
            # Redirigimos al tipo de ropa que hemos detectado
            return redirect(f'similar/{type_of_clothing}/')
    # Si alguno de los pasos ha fallado redirigimos a la página principal
    return redirect('/')

def similar(request, dress_type):
    return render(request, 'resultados.html', {'dress_type': dress_type})