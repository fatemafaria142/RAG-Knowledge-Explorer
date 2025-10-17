from django.shortcuts import render

def streaming_chat_example(request):
    """Serve the streaming chat example page"""
    return render(request, 'streaming_chat.html')