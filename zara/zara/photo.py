from django import forms
import logging

# Clase para insertar en el template y poder recoger la foto
class UploadFileForm(forms.Form):
    # Definimos el campo de la foto
    file = forms.FileField()

def handle_photo(file):
    return "Done"