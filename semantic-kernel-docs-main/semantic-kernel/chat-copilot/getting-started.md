---
title: Get started using Chat Copilot locally
description: Run Chat Copilot locally to see how it works.
author: matthewbolanos
ms.topic: samples
ms.author: mabolan
ms.date: 08/03/2023
ms.service: semantic-kernel
---
# Getting started with Chat Copilot

[!INCLUDE [subheader.md](../includes/pat_large.md)]

Chat Copilot is an AI chat application sample created for educational purposes. It is built on Microsoft Semantic Kernel and has two components:
- A frontend [React web app](https://github.com/microsoft/chat-copilot/tree/main/webapp) that provides a user interface for interacting with the Semantic Kernel.
- A backend [.NET web service](https://github.com/microsoft/chat-copilot/tree/main/webapi) that provides an API for the React web app to interact with the Semantic Kernel.

In this article, we'll walk through the steps to run these two components locally on your machine. The [Chat Copilot reference app](https://github.com/microsoft/chat-copilot/blob/main/README.md) is located in the Chat Copilot GitHub repository.

> [!IMPORTANT]
> Each chat interaction will call Azure OpenAI/OpenAI which will use tokens that you may be billed for.

## Requirements
You will need the following items to run the sample:

> [!div class="checklist"]
> * [Git](https://git-scm.com/book/v2/Getting-Started-Installing-Git)
> * [.NET 7.0 SDK](https://dotnet.microsoft.com/download/dotnet/7.0) _(via Setup script)_
> * [Node.js](https://nodejs.org/en/download) _(via Setup script)_
> * [Yarn](https://classic.yarnpkg.com/docs/install) _(via Setup script)_
> * [Azure account](https://azure.microsoft.com/free)
> * [Azure AD Tenant](/azure/active-directory/develop/quickstart-create-new-tenant)
> * AI Service:

| AI Service   | Requirement                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Azure OpenAI | - [Access](https://aka.ms/oai/access)<br>- [Resource](/azure/ai-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource)<br>- [Deployed models](/azure/ai-services/openai/how-to/create-resource?pivots=web-portal#deploy-a-model) (`gpt-35-turbo` and `text-embedding-ada-002`)<br>- [Endpoint](/azure/ai-services/openai/tutorials/embeddings?tabs=command-line#retrieve-key-and-endpoint)<br>- [API key](/azure/ai-services/openai/tutorials/embeddings?tabs=command-line#retrieve-key-and-endpoint) |
| OpenAI       | - [Account](https://platform.openai.com)<br>- [API key](https://platform.openai.com/account/api-keys)                    

## Instructions
1) Register an application.

    To register an application, follow [these instructions](/azure/active-directory/develop/quickstart-register-app) and use the settings below:

   - `Supported account types`: "_Accounts in any organizational directory (Any Azure AD directory - Multitenant) and personal Microsoft accounts (e.g. Skype, Xbox)_" 
   - `Redirect URI (optional)`: _Single-page application (SPA)_ and use _http://localhost:3000_.
    
    > [!NOTE]
    > Take note of your application's _Application (client) ID_. Chat Copilot will use this ID for authentication.

2) Install .NET 7.0 SDK, Node.js, and Yarn on your machine.

    # [Windows](#tab/Windows)
    Open a PowerShell terminal as an administrator and navigate to the _\scripts_ directory in the Chat Copilot project.

    ```powershell
    cd .\scripts\
    ```

    Next, run the following command to install the required dependencies. This script will also install [Chocolatey](https://chocolatey.org/):
    ```powershell
    .\Install.ps1
    ```

    # [Debian/Ubuntu Linux](#tab/Linux)
    Open a Bash terminal as an administrator and navigate to the _/scripts_ directory in the Chat Copilot project:
    ```bash
    cd ./scripts/

    # Ensure the install scripts are executable
    chmod +x Install-apt.sh
    ```

    Next, run the following command to install the required dependencies:
    ```bash
    ./Install-apt.sh
    ```
    # [macOS](#tab/macos)

    Open a Bash terminal as an administrator and navigate to the _/scripts_ directory in the Semantic Kernel project:
    ```bash
    cd ./scripts/

    # Ensure the install scripts are executable
    chmod +x Install-brew.sh
    ```

    Next, run the following command to install the required dependencies. The macOS install script uses [Homebrew](https://brew.sh/) to install dependencies:
    ```bash
    ./Install-brew.sh
    ```
    ---

3) Configure Chat Copilot.

    # [PowerShell](#tab/Powershell)

    Replace the values in brackets below before running the command:

    ```powershell
    .\Configure.ps1 -AIService {AI_SERVICE} -APIKey {API_KEY} -Endpoint {AZURE_OPENAI_ENDPOINT} -ClientId {AZURE_APPLICATION_ID} 
    ```

    - `AI_SERVICE`: `AzureOpenAI` or `OpenAI`.
    - `API_KEY`: The _API key_ for Azure OpenAI or for OpenAI.
    - `AZURE_OPENAI_ENDPOINT`: The Azure OpenAI resource _Endpoint_ address. Omit `-Endpoint` if using OpenAI.
    - `AZURE_APPLICATION_ID`: The _Application (client) ID_ associated with the registered application.

    - (Optional): To set a specific Tenant Id for the web application, use the parameter:

        ```powershell
        -TenantId {TENANT_ID}
        ```

    > [!IMPORTANT]
    > For **Azure OpenAI**, if you deployed models `gpt-35-turbo` and `text-embedding-ada-002` with custom names (instead of each own's given name), also use the parameters:

    ```powershell
    -CompletionModel {DEPLOYMENT_NAME} -EmbeddingModel {DEPLOYMENT_NAME} -PlannerModel {DEPLOYMENT_NAME}
    ```

    # [Bash](#tab/Bash)
    First, ensure the configuration script is executable:

    ```bash
    # Ensure the configure scripts are executable
    chmod +x Configure.sh
    ```

    Replace the values in brackets below before running the command:

    ```bash
    ./Configure.sh --aiservice {AI_SERVICE} --apikey {API_KEY} --endpoint {AZURE_OPENAI_ENDPOINT} --clientid {AZURE_APPLICATION_ID} 
    ```

    - `AI_SERVICE`: `AzureOpenAI` or `OpenAI`.
    - `API_KEY`: The _API key_ for Azure OpenAI or for OpenAI.
    - `AZURE_OPENAI_ENDPOINT`: The Azure OpenAI resource _Endpoint_ address. Omit `--endpoint` if using OpenAI.
    - `AZURE_APPLICATION_ID`: The _Application (client) ID_ associated with the registered application.

    - (Optional): To set a specific Tenant Id, use the parameter:

        ```bash
        --tenantid {TENANT_ID}
        ```

    > [!IMPORTANT]
    > For **Azure OpenAI**, if you deployed models `gpt-35-turbo` and `text-embedding-ada-002` with custom names (instead of each own's given name), also use the parameters: 

    ```bash 
    --completionmodel {DEPLOYMENT_NAME} --embeddingmodel {DEPLOYMENT_NAME} --plannermodel {DEPLOYMENT_NAME} 
    ```

    ---

4) Run the start script.
    
    # [PowerShell](#tab/Powershell)

    ```powershell
    .\Start.ps1
    ```

    It may take a few minutes for Yarn packages to install on the first run.

    Confirm pop-ups are not bocked and you are logged in with the same account used to register the application. 

    # [Bash](#tab/Bash)

    ```bash
    # Ensure the start scripts are executable
    chmod +x Start.sh
    chmod +x Start-Backend.sh
    chmod +x Start-Frontend.sh

    # Start CopilotChat 
    ./Start.sh
    ```

    It may take a few minutes for Yarn packages to install on the first run.

    Confirm pop-ups are not bocked and you are logged in with the same account used to register the application. 

    ---
5) Congrats! A browser should automatically launch and navigate to _https://localhost:3000_ with the sample app running.

## Next step

Now that you've gotten Chat Copilot running locally, you can learn how to customize it to your needs.

> [!div class="nextstepaction"]
> [Customize Chat Copilot](./customizing-chat-copilot.md)
