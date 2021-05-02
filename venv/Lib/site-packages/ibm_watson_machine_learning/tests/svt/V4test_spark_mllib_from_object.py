import unittest

# SPARK_HOME_PATH = os.environ['SPARK_HOME']
# PYSPARK_PATH = str(SPARK_HOME_PATH) + "/python/"
# sys.path.insert(1, path_join(PYSPARK_PATH))

import logging
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient
from models_preparation import *


class TestWMLClientWithSpark(unittest.TestCase):
    deployment_uid = None
    space_uid = None
    space_href = None
    model_uid = None
    scoring_url = None
    deployment_name = "Test deployment"
    space_name = 'tests_sdk_space'
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
        TestWMLClientWithSpark.space_id = get_space_id(self.wml_client, self.space_name,
                                                       cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
        self.wml_client.set.default_space(self.space_id)

    def test_03_publish_model(self):
        TestWMLClientWithSpark.logger.info("Creating spark model ...")

        model_data = create_spark_mllib_model_data()

        TestWMLClientWithSpark.logger.info("Publishing spark model ...")

        self.wml_client.repository.ModelMetaNames.show()

        sw_spec_id = self.wml_client.software_specifications.get_id_by_name("spark-mllib_2.4")

        model_props = {self.wml_client.repository.ModelMetaNames.NAME: "Spark",
            self.wml_client.repository.ModelMetaNames.TYPE: "mllib_2.4",
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                       }

        published_model = self.wml_client.repository.store_model(model=model_data['model'], meta_props=model_props, training_data=model_data['training_data'], pipeline=model_data['pipeline'])
        TestWMLClientWithSpark.model_uid = self.wml_client.repository.get_model_uid(published_model)
        TestWMLClientWithSpark.logger.info("Published model ID:" + str(TestWMLClientWithSpark.model_uid))
        self.assertIsNotNone(TestWMLClientWithSpark.model_uid)

    def test_04_get_details(self):
        TestWMLClientWithSpark.logger.info("Get details")
        details = self.wml_client.repository.get_details(self.model_uid)
        print(details)
        TestWMLClientWithSpark.logger.debug("Model details: " + str(details))
        self.assertTrue("Spark" in str(details))

    def test_05_create_deployment(self):
        TestWMLClientWithSpark.logger.info("Create deployment")
        deployment = self.wml_client.deployments.create(self.model_uid, meta_props={self.wml_client.deployments.ConfigurationMetaNames.NAME: self.deployment_name,self.wml_client.deployments.ConfigurationMetaNames.ONLINE: {}})


        TestWMLClientWithSpark.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithSpark.logger.debug("Online deployment: " + str(deployment))
        TestWMLClientWithSpark.scoring_url = self.wml_client.deployments.get_scoring_href(deployment)
        TestWMLClientWithSpark.logger.debug("Scoring url: {}".format(TestWMLClientWithSpark.scoring_url))
        TestWMLClientWithSpark.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        TestWMLClientWithSpark.logger.debug("Deployment uid: {}".format(TestWMLClientWithSpark.deployment_uid))
        self.assertTrue("online" in str(deployment))

    def test_06_get_deployment_details(self):
        TestWMLClientWithSpark.logger.info("Get deployment details")
        deployment_details = self.wml_client.deployments.get_details(TestWMLClientWithSpark.deployment_uid)
        print(deployment_details)
        TestWMLClientWithSpark.logger.debug("Deployment details: {}".format(deployment_details))
        self.assertTrue(self.deployment_name in str(deployment_details))

    def test_07_score(self):
        TestWMLClientWithSpark.logger.info("Score the model")
        scoring_data = {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                    "fields": ["GENDER","AGE","MARITAL_STATUS","PROFESSION"],
                    "values": [["M",23,"Single","Student"],["M",55,"Single","Executive"]]
                }
            ]
        }
        predictions = self.wml_client.deployments.score(TestWMLClientWithSpark.deployment_uid, scoring_data)
        print("Predictions: {}".format(predictions))
        self.assertTrue("prediction" in str(predictions))

    def test_08_delete_deployment(self):
        TestWMLClientWithSpark.logger.info("Delete deployment")
        self.wml_client.deployments.delete(TestWMLClientWithSpark.deployment_uid)

    def test_09_delete_model(self):
        TestWMLClientWithSpark.logger.info("Delete model")
        self.wml_client.repository.delete(TestWMLClientWithSpark.model_uid)

if __name__ == '__main__':
    unittest.main()
