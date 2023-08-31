import json
from semantic_kernel import Kernel
from semantic_kernel.skill_definition import (
    sk_function,
)
from semantic_kernel.orchestration.sk_context import SKContext
from plugins.FaqSearchPlugin.FaqSearch import FAQQdrantSearch


class OrchestratorPlugin:
    def __init__(self, kernel: Kernel):
        self._kernel = kernel

    @sk_function(
        description="Routes the request to the appropriate function",
        name="route_request",
    )
    async def RouteRequest(self, context: SKContext) -> str:
        print("OrchestratorPlugin initialized")
        # Save the original user request
        request = context["input"]
        
        # Retrieve the intent from the user request
        # GetIntent = self._kernel.skills.get_function("SemanticSkill", "GetIntent")
        # GetAns = self._kernel.skills.get_function("SemanticSkill", "GetAns")
        
        pluginsDirectory = "./plugins/QueryingPlugin"
        GetIntent = self._kernel.import_semantic_skill_from_directory(
            pluginsDirectory, "GetIntent"
            )
        GetAns = self._kernel.import_semantic_skill_from_directory(
            pluginsDirectory, "GetAns"
            )
        
        await GetIntent.invoke_async(context=context)
        intent = context["input"].strip()

        # Create search faq function
        FAQQuery = self._kernel.skills.get_function("FaqSearchPlugin", "faq_query")
        
        # Create a new context object with the original request
        pipelineContext = self._kernel.create_new_context()
        pipelineContext["query"] = request
        pipelineContext["input"] = request
        
        # Run the functions in a pipeline
        output = await self._kernel.run_async(
            GetIntent,
            FAQQuery,
            # GetAns,
            input_context=pipelineContext,
        )

        return output["input"]