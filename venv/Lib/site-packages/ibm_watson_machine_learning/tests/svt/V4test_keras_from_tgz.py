import unittest
import os
import logging
from preparation_and_cleaning import *
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient


class TestWMLClientWithKeras(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    x_test = None
    space_name = 'tests_sdk_space'
    space_id = None
    model_path = os.path.join('.', 'svt', 'artifacts', 'keras_model.tgz')
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """

        cls.wml_credentials = get_wml_credentials()

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)

        if not cls.wml_client.ICP:
            cls.cos_credentials = get_cos_credentials()
            cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
            cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)
        cls.project_id = cls.wml_credentials.get('project_id')

    def test_00a_space_cleanup(self):
        space_cleanup(self.wml_client,
                      get_space_id(self.wml_client, self.space_name,
                                   cos_resource_instance_id=self.cos_resource_instance_id),
                      days_old=7)
        TestWMLClientWithKeras.space_id = get_space_id(self.wml_client, self.space_name,
                                                       cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
        self.wml_client.set.default_space(self.space_id)

    def test_02_publish_model(self):
        TestWMLClientWithKeras.logger.info("Creating keras model ...")
        from keras.datasets import mnist

        from keras import backend as K

        batch_size = 128
        num_classes = 10
        epochs = 1

        # input shape
        img_rows, img_cols = 28, 28

        # samples to train
        num_train_samples = 500

#        print(K._backend)

        # prepare train and test datasets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        x_test = x_test.astype('float32')
        x_test /= 255
        print(x_test.shape[0], 'test samples')

        TestWMLClientWithKeras.x_test = x_test

        self.logger.info("Publishing keras model ...")

        self.wml_client.repository.ModelMetaNames.show()

        self.wml_client.software_specifications.list()
        sw_spec_id = self.wml_client.software_specifications.get_id_by_name("default_py3.7")

        model_props = {self.wml_client.repository.ModelMetaNames.NAME: "KerasModel-gzip",
            self.wml_client.repository.ModelMetaNames.TYPE: "tensorflow_2.1",
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                       }
        published_model_details = self.wml_client.repository.store_model(model=TestWMLClientWithKeras.model_path,
                                                                     meta_props=model_props)  # , training_data=digits.data, training_target=digits.target)
        TestWMLClientWithKeras.model_uid = self.wml_client.repository.get_model_uid(published_model_details)
        TestWMLClientWithKeras.model_url = self.wml_client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithKeras.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithKeras.model_url))
        self.assertIsNotNone(TestWMLClientWithKeras.model_uid)
        self.assertIsNotNone(TestWMLClientWithKeras.model_url)

    # def test_03_download_model(self):
    #     TestWMLClientWithKeras.logger.info("Download model")
    #     try:
    #         os.remove('download_test_url')
    #     except OSError:
    #         pass
    #
    #     try:
    #         file = open('download_test_uid', 'r')
    #     except IOError:
    #         file = open('download_test_uid', 'w')
    #         file.close()
    #
    #     self.wml_client.repository.download(TestWMLClientWithKeras.model_uid, filename='download_test_url')
    #     self.assertRaises(WMLClientError, self.wml_client.repository.download, TestWMLClientWithKeras.model_uid,
    #                       filename='download_test_uid')

    def test_04_get_details(self):
        TestWMLClientWithKeras.logger.info("Get model details")
        details = self.wml_client.repository.get_details(self.model_uid)
        TestWMLClientWithKeras.logger.debug("Model details: " + str(details))
        self.assertTrue("KerasModel-gzip" in str(details))

    def test_05_create_deployment(self):
        TestWMLClientWithKeras.logger.info("Create deployments")
        deployment = self.wml_client.deployments.create(TestWMLClientWithKeras.model_uid, meta_props={
            self.wml_client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.wml_client.deployments.ConfigurationMetaNames.ONLINE:{}})
        TestWMLClientWithKeras.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithKeras.logger.info("Online deployment: " + str(deployment))
        self.assertTrue(deployment is not None)
        TestWMLClientWithKeras.scoring_url = self.wml_client.deployments.get_scoring_href(deployment)
        TestWMLClientWithKeras.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))

    def test_06_get_deployment_details(self):
        TestWMLClientWithKeras.logger.info("Get deployment details")
        deployment_details = self.wml_client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

    def test_07_get_deployment_details_using_uid(self):
        TestWMLClientWithKeras.logger.info("Get deployment details using uid")
        deployment_details = self.wml_client.deployments.get_details(TestWMLClientWithKeras.deployment_uid)
        self.assertIsNotNone(deployment_details)

    def test_08_score(self):
        TestWMLClientWithKeras.logger.info("Score model")
        x_score_1 = TestWMLClientWithKeras.x_test[23].tolist()
        x_score_2 = TestWMLClientWithKeras.x_test[32].tolist()
        scoring_payload = {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA : [
                {
                    'values': [x_score_1, x_score_2]
                }
            ]
        }
        predictions = self.wml_client.deployments.score(TestWMLClientWithKeras.deployment_uid, scoring_payload)
        self.assertTrue("prediction" in str(predictions))

    def test_09_delete_deployment(self):
        TestWMLClientWithKeras.logger.info("Delete deployment")
        self.wml_client.deployments.delete(TestWMLClientWithKeras.deployment_uid)

    def test_10_delete_model(self):
        TestWMLClientWithKeras.logger.info("Delete model")
        self.wml_client.repository.delete(TestWMLClientWithKeras.model_uid)


if __name__ == '__main__':
    unittest.main()
