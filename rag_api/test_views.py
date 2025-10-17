from django.shortcuts import render

def test_api_view(request):
    """
    View to render the API testing interface
    """
    return render(request, 'test_api.html')