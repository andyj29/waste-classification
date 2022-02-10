from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import ParseError
from rest_framework import status
from .models import Image
from .services import classify_image


class ClassifyImage(APIView):
    def post(self, request, *args, **kwargs):
        try:
            file = self.request.data['file']
        except KeyError:
            raise ParseError('Request has no image file')
        image = Image.objects.create(image=file)
        prediction = classify_image(image.get_url)
        data = {'label': prediction}
        return Response(data, status=status.HTTP_202_ACCEPTED)
