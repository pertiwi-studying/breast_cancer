from django.shortcuts import render


# our home page view
def home(request):
    return render(request, "base.html")


# custom method for generating predictions
def getPredictions(
    mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness
):
    import pickle

    model = pickle.load(open("ml/ml_model.sav", "rb"))
    scaled = pickle.load(open("ml/scaler.sav", "rb"))
    prediction = model.predict(
        scaled.transform(
            [[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]]
        )
    )

    if prediction == 0:
        return "not diagnosed with breast cancer"
    elif prediction == 1:
        return "breast cancer diagnosed"
    else:
        return "error"


# our result page view
def result(request):
    mean_radius = float(request.GET["mean_radius"])
    mean_texture = float(request.GET["mean_texture"])
    mean_perimeter = float(request.GET["mean_perimeter"])
    mean_area = float(request.GET["mean_area"])
    mean_smoothness = float(request.GET["mean_smoothness"])

    result = getPredictions(
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness
    )

    return render(request, "result.html", {"result": result})
