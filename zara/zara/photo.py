from zara.models.type_model import dress_type
from django import forms

# Clase para insertar en el template y poder recoger la foto
class UploadFileForm(forms.Form):
    # Definimos el campo de la foto
    file = forms.FileField()

# Buscamos todas las fotos similares
def handle_photo(file):
    return dress_type(file)