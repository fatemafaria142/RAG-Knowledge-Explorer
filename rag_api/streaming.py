import json
from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.exceptions import ValidationError
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from .models import RAGSession, ChatMessage
from .serializers import ChatRequestSerializer, ChatMessageSerializer
from .rag_system import get_rag_system


from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

class StreamingChatView(APIView):
    """Handle streaming chat messages with real-time response generation"""
    
    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['session_id', 'message'],
            properties={
                'session_id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='The ID of the RAG session'),
                'message': openapi.Schema(type=openapi.TYPE_STRING, description='The message/question to process')
            }
        ),
        operation_description="Send a message to an existing RAG session with streaming response",
        responses={
            status.HTTP_200_OK: "Streaming response (text/event-stream)",
            status.HTTP_400_BAD_REQUEST: "Invalid input",
            status.HTTP_404_NOT_FOUND: "Session not found",
            status.HTTP_500_INTERNAL_SERVER_ERROR: "Processing error"
        },
        tags=['Chat']
    )
    def post(self, request):
        # Validate the request
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return StreamingHttpResponse(
                json.dumps({"error": serializer.errors}),
                content_type="application/json",
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Extract data
        session_id = serializer.validated_data['session_id']
        message_content = serializer.validated_data['message']
        
        try:
            # Get session
            session = get_object_or_404(RAGSession, id=session_id)
            
            # Get chat history
            chat_history = ChatMessage.objects.filter(session=session).order_by('timestamp')
            history_data = ChatMessageSerializer(chat_history, many=True).data
            
            # Save user message
            user_message = ChatMessage.objects.create(
                session=session,
                message_type='user',
                content=message_content
            )
            
            # Use session values for instructions and description
            instructions = session.instructions
            description = session.description
            
            # Combine instructions and description if available
            combined_instructions = ""
            if instructions and description:
                combined_instructions = f"Description: {description}\n\nInstructions: {instructions}"
            elif instructions:
                combined_instructions = instructions
            elif description:
                combined_instructions = f"Description: {description}"
            
            # Get the RAG system
            rag_system = get_rag_system()
            if rag_system is None:
                # Return an error response as a streaming event
                return StreamingHttpResponse(
                    streaming_content=(f'data: {json.dumps({"error": "RAG system initialization failed", "done": True})}\n\n',),
                    content_type='text/event-stream'
                )
            
            # Create a variable to store the complete response
            complete_response = None
            sources = []
            
            # Set to keep track of finished chunks 
            # (we need to save the message after streaming is complete)
            is_complete = False
            
            def stream_response():
                nonlocal complete_response, sources, is_complete
                
                # Stream the response
                try:
                    print(f"Starting streaming chat for session {session_id}")
                    print(f"Message: {message_content}")
                    print(f"Instructions length: {len(combined_instructions) if combined_instructions else 0}")
                    print(f"Chat history: {len(history_data)} messages")
                    
                    # Verify RAG system state
                    print(f"LLM Client initialized: {rag_system.llm_client is not None}")
                    print(f"Embedding Client initialized: {rag_system.embed_client is not None}")
                    print(f"LLM Deployment: {rag_system.llm_deployment}")
                    
                    # Get the generator from the chat method
                    stream_generator = rag_system.chat(
                        session_id=str(session_id),
                        question=message_content,
                        chat_history=history_data,
                        instructions=combined_instructions,
                        stream=True
                    )
                    
                    print(f"Generator created successfully: {stream_generator is not None}")
                    
                    # Process each chunk from the generator
                    chunk_count = 0
                    for chunk in stream_generator:
                        chunk_count += 1
                        if not isinstance(chunk, dict):
                            print(f"Warning: Received non-dict chunk: {type(chunk)}: {chunk}")
                            continue
                            
                        # Update the complete response
                        if "full_response" in chunk:
                            complete_response = chunk["full_response"]
                        
                        # Update sources
                        if "sources" in chunk:
                            sources = chunk.get("sources", [])
                        
                        # Mark as complete if this is the last chunk
                        if chunk.get("done", False):
                            is_complete = True
                            print("Final chunk received (done=True)")
                        
                        # Debug log every 10th chunk to avoid excessive logging
                        if chunk_count % 10 == 0:
                            print(f"Processing chunk #{chunk_count}, response length: {len(complete_response) if complete_response else 0}")
                        
                        # Yield the chunk as a server-sent event
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    print(f"Streaming completed: {chunk_count} chunks processed")
                    
                    # Save the assistant message when complete
                    try:
                        # If we have a complete response, use it
                        # Otherwise use a default message
                        message_to_save = complete_response
                        if not message_to_save:
                            message_to_save = "I apologize, but I couldn't generate a proper response. Please try again."
                            print(f"No complete response was received, saving default message")
                        
                        # Create the chat message in the database
                        ChatMessage.objects.create(
                            session=session,
                            message_type='assistant',
                            content=message_to_save,
                            sources=sources
                        )
                        
                        # Update session timestamp
                        session.save()
                        print(f"Saved assistant message of length: {len(message_to_save)}")
                    except Exception as e:
                        print(f"Error saving chat message: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Still continue - the message may have been delivered to the client
                    
                except Exception as e:
                    import traceback
                    print("===== STREAMING CHAT ERROR =====")
                    print(f"Session ID: {session_id}")
                    print(f"Message: {message_content}")
                    print(f"Error: {str(e)}")
                    traceback.print_exc()
                    print("================================")
                    
                    error_msg = f"Failed to process streaming chat: {str(e)}"
                    yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"
            
            # Return a streaming response
            return StreamingHttpResponse(
                streaming_content=stream_response(),
                content_type='text/event-stream'
            )
            
        except Exception as e:
            return StreamingHttpResponse(
                json.dumps({"error": f"Failed to process streaming chat: {str(e)}"}),
                content_type="application/json",
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )