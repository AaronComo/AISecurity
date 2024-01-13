import json

from PIL import Image
from django.http import HttpResponse, HttpResponseForbidden
from django.middleware.csrf import get_token
from django.shortcuts import render, redirect

from robustness.eval import test_robust
from robustness.forms import UploadForm, ResponseForm
from .eval import adv_attack

HTTP_PREFIX = "http://127.0.0.1"
PORT = "8000"


def index(request):
    return render(request, 'robustness/index.html')


def uploadImg(request):
    """
    robustness:
    获取鲁棒性评估页面，接收参数并返回评估结果
    """
    if request.method == 'POST':
        af = UploadForm(request.POST, request.FILES)
        print(request.POST)
        if af.is_valid():
            print("Handling uploaded image.")
            model = af.cleaned_data['model']
            # y_true = af.cleaned_data['label']
            img = af.cleaned_data['img']
            y_true = img.name.split('.')[0]
            print(f'model:{model}')
            print(f'y_true:{y_true}')
            print(f'img:{img}')
            if y_true[0] != 'n':
                return HttpResponseForbidden("Error file name. Please rename it to match ImageNet.")
            # Img = IMG(img=img)
            # Img.save()
            handle_uploaded_file(img)
            x = './static/robustness/' + str(img)
            pth, pth1, true, pre = adv_attack(str(model), x, y_true)

            # Set return values and some status variables
            cont = {
                "ori_path": x,
                "adv_path": pth,
                "adversarial": pth1,
                "model": model,
                "true": true,
                "pre": pre,
            }
            # request.session["uploaded"] = True
            # request.session["variables"] = cont
            # return render(request, 'robustness/uploadimg.html', context=cont)
            print("uploadImg: Done.")
            return HttpResponse(json.dumps(wrapper(cont, method="encode"), ensure_ascii=False))
    else:
        request.session.flush()
        af = UploadForm()
        return render(request, 'robustness/uploadimg.html', context={'af': af})


def testRobustness(request):
    r"""
    Test given AI model.
    """
    if request.method == "POST":
        print('starting robustness test...')
        print(request.body)
        cont = str(request.body)
        cont = wrapper(json.loads(cont.split('\'')[1]), method="decode")
        # cont = request.session["variables"]
        print(cont)
        if cont == {}:
            return HttpResponseForbidden('Empty request.')
        advImage = Image.open(cont["adversarial"])
        # advImage.show()
        pth2, pre = test_robust(cont["model"], advImage)
        cont.update({
            "robust_path": pth2,
            "pre": pre
        })
        print('testRobustness: Done.')
        # return render(request, 'robustness/showimg.html', cont)
        return HttpResponse(json.dumps(wrapper(cont, method="encode"), ensure_ascii=False))
    return redirect("/robustness")


def handle_uploaded_file(f):
    with open(f"./static/robustness/{f}", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def addr(st, prefix):
    s = st.split('/')
    # print(s)
    return prefix + s[len(s) - 1]


def wrapper(cont, method) -> dict:
    r"""
    method = ["encode", "decode"]
    encode: relative -> http
    decode: http -> relative
    """
    if method == "encode":
        prefix = f"{HTTP_PREFIX}:{PORT}/static/robustness/"
    else:
        prefix = './static/robustness/'
    for i in cont.keys():
        if i in ["model", "true", "pre"]:
            continue
        st = cont[i]
        cont[i] = addr(st, prefix)
    return cont


def token(request):
    token = get_token(request)
    return HttpResponse(json.dumps({"token": token}))
