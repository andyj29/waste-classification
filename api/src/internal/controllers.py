from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import ParseError, NotFound
from rest_framework import status
from .serializers import LocationSerializer, WasteCategorySerializer
from .models import Image, Location, WasteCategory
from .services import classify_image, get_area_from_lat_long, is_in_gta


class ClassifyImage(APIView):
    def post(self, request, *args, **kwargs):

        try:
            file = self.request.data['file']
        except KeyError:
            raise ParseError('Request has no image file')

        image = Image.objects.create(image=file)

        prediction = classify_image(image.get_url)

        category = WasteCategory.objects.filter(type=prediction['label']).first()

        category_serializer = WasteCategorySerializer(category)

        try:
            latitude = self.request.data['lat']
            longitude = self.request.data['long']
        except KeyError:
            raise ParseError('Request has no longitude/latitude')
        
        location = get_area_from_lat_long(latitude, longitude)

        if is_in_gta(location):
            queryset = Location.objects.filter(area=location, category=category)
            loc_list = LocationSerializer(queryset, many=True)
        else:
            raise NotFound({
                'prediction': prediction,
                'category': category_serializer.data,
                'locations':'Our location service only works inside the GTA'
            })

        data = {
            'prediction': prediction,
            'category': category_serializer.data,
            'locations': loc_list.data
        }

        return Response(data, status=status.HTTP_202_ACCEPTED)
