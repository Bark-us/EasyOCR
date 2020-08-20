import torch
from ts.torch_handler.base_handler import BaseHandler
import io


class CraftHandler(BaseHandler):

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

    def postprocess(self, data):
        """
        Override to customize the post-processing
        :param data: Torch tensor, containing prediction output from the model
        :return: Python list
        """
        return data.tolist()