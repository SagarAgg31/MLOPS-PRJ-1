pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }
    
    stages {
        stage('Cloning Github Repo') {
            steps {
                script {
                    echo 'Cloning Github repository...'
                    checkout scmGit(branches: [[name: 'main']], 
                        userRemoteConfigs: [[
                            credentialsId: 'github-token', 
                            url: 'https://github.com/SagarAgg31/MLOPS-PRJ-1.git'
                        ]]
                    )
                }
            }
        }

        stage('Setting Up Virtual Environment & Installing Dependencies') {
            steps {
                script {
                    echo 'Setting up virtual environment and installing dependencies...'
                    sh '''
                        python -m venv ${VENV_DIR}
                        if [ -f "${VENV_DIR}/bin/activate" ]; then
                            . ${VENV_DIR}/bin/activate
                        else
                            call ${VENV_DIR}\\Scripts\\activate
                        fi
                        pip install --upgrade pip
                        pip install -e .
                    '''
                }
            }
        }
    }
}
