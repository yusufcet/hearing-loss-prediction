import unittest
import os
import json
import logging
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient


class TestWMLClientWithSpark(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    definition_url = None
    space_name = 'tests_sdk_space'
    model_meta = None
    pipeline_meta = None
    model_path = os.path.join('.', 'svt', 'artifacts', 'heart-drug-sample', 'drug-selection-model.tgz')
    pipeline_path = os.path.join('.', 'svt', 'artifacts', 'heart-drug-sample', 'drug-selection-pipeline.tgz')
    meta_path = os.path.join('.', 'svt', 'artifacts', 'heart-drug-sample', 'drug-selection-meta.json')
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

        with open(TestWMLClientWithSpark.meta_path) as json_data:
            metadata = json.load(json_data)

        TestWMLClientWithSpark.model_meta = metadata['model_meta']
        TestWMLClientWithSpark.pipeline_meta = metadata['pipeline_meta']

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


    def test_3_publish_model(self):
        TestWMLClientWithSpark.logger.info("Publishing spark model ...")
        self.wml_client.repository.ModelMetaNames.show()

        sw_spec_id = self.wml_client.software_specifications.get_id_by_name("spark-mllib_2.4")

        model_props = {
            self.wml_client.repository.ModelMetaNames.NAME: "SparkModel-from-tar",
            self.wml_client.repository.ModelMetaNames.TYPE: "mllib_2.4",
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
            self.wml_client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES:
                [
                    {
                        'type': 's3',
                        'connection': {
                            'endpoint_url': 'not_applicable',
                            'access_key_id': 'not_applicable',
                            'secret_access_key': 'not_applicable'
                        },
                        'location': {
                            'bucket': 'not_applicable'
                        },
                        'schema': {
                            'id': '1',
                            'type': 'struct',
                            'fields': [{
                                'name': 'AGE',
                                'type': 'float',
                                'nullable': True,
                                'metadata': {
                                    'modeling_role': 'target'
                                }
                            }, {
                                'name': 'SEX',
                                'type': 'string',
                                'nullable': True,
                                'metadata': {}
                            }, {
                                'name': 'CHOLESTEROL',
                                'type': 'string',
                                'nullable': True,
                                'metadata': {}
                            }, {
                                'name': 'BP',
                                'type': 'string',
                                'nullable': True,
                                'metadata': {}
                            }, {
                                'name': 'NA',
                                'type': 'float',
                                'nullable': True,
                                'metadata': {}
                            }, {
                                'name': 'K',
                                'type': 'float',
                                'nullable': True,
                                'metadata': {}
                            }]
                        }
                    }
                    ]}


        print('XXX' + str(model_props))
        published_model = self.wml_client.repository.store_model(model=self.model_path, meta_props=model_props)
        print("Model details: " + str(published_model))

        TestWMLClientWithSpark.model_uid = self.wml_client.repository.get_model_uid(published_model)
        TestWMLClientWithSpark.logger.info("Published model ID:" + str(TestWMLClientWithSpark.model_uid))
        self.assertIsNotNone(TestWMLClientWithSpark.model_uid)

    def test_4_get_details(self):
        TestWMLClientWithSpark.logger.info("Get model details")
        details = self.wml_client.repository.get_details(self.model_uid)
        print(details)
        TestWMLClientWithSpark.logger.debug("Model details: " + str(details))
        self.assertTrue("SparkModel" in str(details))

    def test_5_create_deployment(self):
        TestWMLClientWithSpark.logger.info("Create deployment")
        deployment = self.wml_client.deployments.create(self.model_uid, meta_props={self.wml_client.deployments.ConfigurationMetaNames.NAME: "best-drug model deployment",self.wml_client.deployments.ConfigurationMetaNames.ONLINE: {}})
        TestWMLClientWithSpark.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithSpark.logger.debug("Online deployment: " + str(deployment))
        TestWMLClientWithSpark.scoring_url = self.wml_client.deployments.get_scoring_href(deployment)
        TestWMLClientWithSpark.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))

    def test_6_get_deployment_details(self):
        TestWMLClientWithSpark.logger.info("Get deployment details")
        deployment_details = self.wml_client.deployments.get_details(TestWMLClientWithSpark.deployment_uid)
        print(deployment_details)
        self.assertTrue('best-drug model deployment' in str(deployment_details))

    def test_6_score(self):
        TestWMLClientWithSpark.logger.info("Score the model")
        scoring_data = {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                    "fields": ["AGE", "SEX", "BP", "CHOLESTEROL", "NA", "K"],
                    "values": [[20.0, "F", "HIGH", "HIGH", 0.71, 0.07], [55.0, "M", "LOW", "HIGH", 0.71, 0.07]]
                }
            ]
        }

        predictions = self.wml_client.deployments.score(TestWMLClientWithSpark.deployment_uid, scoring_data)
        print(predictions)
        self.assertTrue("predictedLabel" in str(predictions))

    def test_7_delete_deployment(self):
        TestWMLClientWithSpark.logger.info("Delete deployment")
        self.wml_client.deployments.delete(TestWMLClientWithSpark.deployment_uid)

    def test_8_delete_model(self):
        TestWMLClientWithSpark.logger.info("Delete model")
        self.wml_client.repository.delete(TestWMLClientWithSpark.model_uid)


if __name__ == '__main__':
    unittest.main()
