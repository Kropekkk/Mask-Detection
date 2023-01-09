from django.shortcuts import render
from .forms import ImageUploadForm
from .detect import predict

def index(request):
    path_img = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()

            try:
                path_img = predict(image_bytes)
                
            except RuntimeError as re:
                print(re)

            pass
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'image_uri': path_img,
    }

    return render(request, 'myapp/index.html', context)
