import torch
from ts.torch_handler.base_handler import BaseHandler
import io


class TextHandler(BaseHandler):

    def preprocess(self, data):
        """
        Override to customize the pre-processing
        :param data: Python list of data items
        :return: input tensor on a device
        """
        byte_data = io.BytesIO(data[0]['body'])
        tensor = torch.load(byte_data)
        byte_data.close()
        return tensor

    def handle(self, data, context):
        """
        Entry point for default handler
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        self.context = context

        image_tensor = self.preprocess(data)
        data = self.inference(image_tensor)
        data = self.postprocess(data)
        return data

    def postprocess(self, data):
        """
        Override to customize the post-processing
        :param data: Torch tensor, containing prediction output from the model
        :return: Python list
        """
        buffer = io.BytesIO()
        torch.save(data.unsqueeze(0), buffer)
        return_data = [buffer.getvalue()]
        return return_data
