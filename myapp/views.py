from django.shortcuts import render
from .forms import ImageUploadForm

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            #transform image and make predicitions
            pass
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
    }
    return render(request, 'app/index.html', context)
