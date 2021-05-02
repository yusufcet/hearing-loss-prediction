import unittest
import os
import logging
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient


class TestWMLClientWithPMML(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    space_id = None
    space_name = 'tests_sdk_space'
    model_path = os.path.join('.', 'svt', 'artifacts', 'iris_chaid.xml')
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
        TestWMLClientWithPMML.space_id = get_space_id(self.wml_client, self.space_name,
                                                      cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
        self.wml_client.set.default_space(self.space_id)

    def test_02_publish_model(self):

        self.logger.info("Publishing PMML model ...")

        self.wml_client.repository.ModelMetaNames.show()

        self.wml_client.software_specifications.list()
        sw_spec_id = self.wml_client.software_specifications.get_id_by_name("spark-mllib_2.3")

        model_props = {self.wml_client.repository.ModelMetaNames.NAME: "LOCALLY created iris prediction model",
            self.wml_client.repository.ModelMetaNames.TYPE: "pmml_4.2.1",
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                       }
        published_model_details = self.wml_client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestWMLClientWithPMML.model_uid = self.wml_client.repository.get_model_uid(published_model_details)
        TestWMLClientWithPMML.model_url = self.wml_client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithPMML.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithPMML.model_url))
        self.assertIsNotNone(TestWMLClientWithPMML.model_uid)
        self.assertIsNotNone(TestWMLClientWithPMML.model_url)

    # def test_03_download_model(self):
    #     try:
    #         os.remove('download_test_url')
    #     except OSError:
    #         pass
    #
    #     self.wml_client.repository.download(TestWMLClientWithPMML.model_uid, filename='download_test_url')
    #     self.assertRaises(WMLClientError, self.wml_client.repository.download, TestWMLClientWithPMML.model_uid, filename='download_test_url')

    def test_04_publish_model_details(self):
        details = self.wml_client.repository.get_details(self.model_uid)
        print(details)
        TestWMLClientWithPMML.logger.debug("Model details: " + str(details))
        self.assertTrue("LOCALLY created iris prediction model" in str(details))

        details_models = self.wml_client.repository.get_model_details()
        TestWMLClientWithPMML.logger.debug("All models details: " + str(details_models))
        self.assertTrue("LOCALLY created iris prediction model" in str(details_models))

    def test_05_create_deployment(self):
        deployment = self.wml_client.deployments.create(artifact_uid=TestWMLClientWithPMML.model_uid,meta_props={self.wml_client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.wml_client.deployments.ConfigurationMetaNames.ONLINE:{}})
        TestWMLClientWithPMML.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithPMML.logger.info("Online deployment: " + str(deployment))
        self.assertTrue(deployment is not None)
        TestWMLClientWithPMML.scoring_url = self.wml_client.deployments.get_scoring_href(deployment)
        TestWMLClientWithPMML.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))

    def test_06_get_deployment_details(self):
        deployment_details = self.wml_client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

    def test_07_get_deployment_details_using_uid(self):
        deployment_details = self.wml_client.deployments.get_details(TestWMLClientWithPMML.deployment_uid)
        print(deployment_details)
        self.assertIsNotNone(deployment_details)

    def test_08_score(self):
        scoring_data = {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA:  [
                {
                    'fields': ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'],
                    'values': [[5.1, 3.5, 1.4, 0.2]]
                }
            ]
        }
        predictions = self.wml_client.deployments.score(TestWMLClientWithPMML.deployment_uid, scoring_data)
        print(predictions)
        predictions_fields = len(predictions)
        self.assertTrue(predictions_fields>0)

    def test_09_delete_deployment(self):
        self.wml_client.deployments.delete(TestWMLClientWithPMML.deployment_uid)

    def test_10_delete_model(self):
        self.wml_client.repository.delete(TestWMLClientWithPMML.model_uid)

if __name__ == '__main__':
    unittest.main()
