---
title: How to write prompts in Semantic Kernel
description: How to write prompts in Semantic Kernel
author: johnmaeda
ms.topic: creating-chains
ms.author: johnmaeda
ms.date: 02/07/2023
ms.service: semantic-kernel
---
# Writing prompts in Semantic Kernel

[!INCLUDE [pat_large.md](../includes/pat_large.md)]

To write an LLM AI prompt that Semantic Kernel is uniquely fit for, all you need is a concrete goal in mind — something you would like an AI to get done for you. For example:

> I want to make a cake. Give me the best chocolate cake recipe you can think of.

Congratulations! You have imagined a delicious ask for Semantic Kernel to run to completion. This ask can be given to the Planner to get decomposed into steps. Although to make the Planner work reliably, you'll need to use the most advanced model available to you. So let's start from writing basic prompts to begin with.



> [!Note]
> Skills are currently being renamed to plugins. This article has been updated to reflect the latest terminology, but some images and code samples may still refer to skills.

> [!TIP]
> Want to easily follow along as you write your first prompts? Download the [Semantic Kernel VS Code Extension](../vs-code-tools/index.md) which allows you to easily create and run prompts from within VS Code.
> [:::image type="content" source="../media/semantic-kernel-tools-install.png" alt-text="Semantic Kernel Tools Extension":::](https://marketplace.visualstudio.com/items?itemName=ms-semantic-kernel.semantic-kernel)


## Writing a simple prompt

Writing prompts is like making a wish. Let's imagine we are entrepreneurs trying to make it in downtown Manhattan and we need to drive more leads to our store. We write the prompt:

```Plain-Prompt
Write me a marketing slogan for my apparel shop in New 
York City with a focus on how affordable we are without 
sacrificing quality.
```

The result of this prompt from an actual LLM AI model is:

```Response-From-LLM-AI-Model
New York Style, Low-Cost Smile: 
Shop at NYC's Best Apparel Store!
```

Let's try another example where we are eager to play with the summarizing capability of LLM AIs and want to show off its superpower when applied to text that we explicitly define:

```Plain-Prompt
Summarize the following text in two sentences or less. 
---Begin Text---
Jan had always wanted to be a writer, ever since they 
were a kid. They spent hours reading books, writing 
stories, and imagining worlds. They grew up and pursued 
their passion, studying literature and journalism, and 
submitting their work to magazines and publishers. They 
faced rejection after rejection, but they never gave up 
hope. Jan finally got their breakthrough, when a famous 
editor discovered their manuscript and offered them a 
book deal.
---End Text---
```

The result of this prompt from an actual LLM AI model is:

```Response-From-LLM-AI-Model
A possible summary is:

Jan's lifelong dream of becoming a writer came true 
when a famous editor offered them a book deal, after 
years of rejection and perseverance.
```

And there we have it. Two simple prompts that aren't asking the model for too much: 1/ we're asking the model to give us a marketing slogan, and separately 2/ we're asking the model to summarize a body of text down to two sentences.

Both of these simple prompts qualify as "functions" that can be packaged as part of an [Semantic Kernel plugin](../create-plugins/index.md). The only problem is that they can do only one thing — as defined by the prompt — and with no flexibility. We set up the first plain prompt in Semantic Kernel within a directory named `SloganMaker` into a file named `skprompt.txt`:

```SloganMaker/skprompt.txt
Write me a marketing slogan for my apparel shop in New 
York City with a focus on how affordable we are without 
sacrificing quality.
```

Similarly, we place the second plain prompt into a directory named `SummarizeBlurb` as a file named into a file named `skprompt.txt`. 

```SummarizeBlurb/skprompt.txt
Summarize the following text in two sentences or less. 
---Begin Text---
Jan had always wanted to be a writer, ever since they 
were a kid. They spent hours reading books, writing 
stories, and imagining worlds. They grew up and pursued 
their passion, studying literature and journalism, and 
submitting their work to magazines and publishers. They 
faced rejection after rejection, but they never gave up 
hope. Jan finally got their breakthrough, when a famous 
editor discovered their manuscript and offered them a 
book deal.
---End Text---
```

Each of these directories comprise a Semantic Kernel function. When both of the directories are placed inside an enclosing directory called `TestPlugin` the result is a brand new plugin. 

```Semantic-Plugins-And-Their-Functions
TestPlugin
│
└─── SloganMaker
|    |
│    └─── skprompt.txt
│    └─── [config.json](../howto/configuringfunctions)
│   
└─── SummarizeBlurb
     |
     └─── skprompt.txt
     └─── [config.json](../howto/configuringfunctions)
```

This plugin can do one of two things by calling one of its two functions:

* `TestPlugin.SloganMaker()` generates a slogan for a specific kind of shop in NYC
* `TestPlugin.SummmarizeBlurb()` creates a short summary of a specific blurb

Next, we'll show you how to make a more powerful plugin by introducing Semantic Kernel prompt templates. But before we do so, you may have noticed the _config.json_ file. That's a special file for customizing how you want the function to run so that its performance can be tuned. If you're eager to know what's inside that file you can go [here](/semantic-kernel/howto/configuringfunctions) but no worries — you'll be running in no time. So let's keep going!

## Writing a more powerful "templated" prompt

Let's say we want to go into the advertising business with AI powering the slogan-side of our offerings. We'd like to encapsulate how we create slogans to be repeatable and across any industry. To do so, we take our first prompt and write it
as such as a "templated prompt":

```SloganMakerFlex/skprompt.txt
Write me a marketing slogan for my {{$INPUT}} in New 
York City with a focus on how affordable we are without 
sacrificing quality.
```

Such "templated" prompts include variables and function calls that can dynamically change the content and the behavior of an otherwise plain prompt. Prompt templates can help you to generate more diverse, relevant, and effective prompts, and to reuse and combine them for different tasks and domains. 

In a templated prompt, the double `{{` curly braces `}}` signify to Semantic Kernel that there's something special for it to notice within the LLM AI prompt. To pass an input to a prompt, we refer to the default input variable `$INPUT` — and by the same token if we have other variables to work with, they will start with a dollar sign `$` as well. 

Our other plain prompt for summarizing text into two sentences can take an `input` by simply replacing the existing body of text and replacing it with `$input` as follows:

```SummarizeBlurbFlex/skprompt.txt
Summarize the following text in two sentences or less. 
---Begin Text---
{{$INPUT}}
---End Text---
```

We can name these two functions `SloganMakerFlex` and `SummarizeBlurbFlex` — as two new Semantic Kernel functions that can belong to a new `TestPluginFlex` plugin that now takes an input. To package these two function to be used by Semantic Kernel in the context of a plugin, we arrange our file hierarchy the same as we did before:

```File-Structure-For-Plugin-Definition-With-Functions
TestPluginFlex
│
└─── SloganMakerFlex
|    |
│    └─── skprompt.txt
│    └─── config.json
│   
└─── SummarizeBlurbFlex
     |
     └─── skprompt.txt
     └─── config.json
```

Recall that the difference between our new "flex" plugins and our original "plain" plugins is that we've gained the added flexibility of being able to pass a single parameter like:

* `TestPluginFlex.SloganMakerFlex('detective agency')` generates a slogan for a 'detective agency' in NYC
* `TestPluginFlex.SummarizeBlurbFlex('<insert long text here>')` creates a short summary of a given blurb

Templated prompts can be further customized beyond a single `$INPUT` variable to take on more inputs to gain even greater flexibility. For instance, if we wanted our SloganMaker plugin to not only take into account the kind of business but also the business' location and specialty, we would write the function as:

```SloganMakerFlex/skprompt.txt
Write me a marketing slogan for my {{$INPUT}} in {{$CITY}} with 
a focus on {{$SPECIALTY}} we are without sacrificing quality.
```

Note that although the use of `$INPUT` made sense as a generic input for a templated prompt, you're likely to want to give it a name that makes immediate sense like `$BUSINESS` — so let's change the function accordingly:

```SloganMakerFlex/skprompt.txt
Write me a marketing slogan for my {{$BUSINESS}} in {{$CITY}} with 
a focus on {{$SPECIALTY}} we are without sacrificing quality.
```

We can replace our `TestPluginFlex` plugin with this new definition for `SloganMakerFlex` to serve the minimum capabilities of a copywriting agency.

In Semantic Kernel, we refer to prompts and templated prompts as _functions_ to clarify their role as a fundamental unit of computation within the kernel. We specifically refer to _semantic_ functions when LLM AI prompts are used; and when conventional programming code is used we say _native_ functions. To learn how to make a native function you can skip ahead to [building a native functions](../create-chains/native-functions.md) if you're anxious.

## Get your kernel ready

First off, you'll want to create an instance of the kernel and configure it to run with Azure OpenAI or regular OpenAI. If you're using Azure OpenAI:

```csharp
using Microsoft.SemanticKernel;

var kernel = Kernel.Builder.Build();

kernel.Config.AddAzureOpenAITextCompletion(
    "Azure_davinci",                        // LLM AI model alias
    "text-davinci-003",                     // Azure OpenAI *Deployment ID*
    "https://contoso.openai.azure.com/",    // Azure OpenAI *Endpoint*
    "...your Azure OpenAI Key..."           // Azure OpenAI *Key*
);
```

If you're using regular OpenAI:

```csharp
using Microsoft.SemanticKernel;

var kernel = Kernel.Builder.Build();

kernel.Config.AddOpenAITextCompletion(
    "OpenAI_davinci",                       // LLM AI model alias
    "text-davinci-003",                     // OpenAI Model Name
    "...your OpenAI API Key...",            // OpenAI API key
    "...your OpenAI Org ID..."              // *optional* OpenAI Organization ID
);
```

## Invoking a semantic function from C#

When running a semantic function from your app's root source directory `MyAppSource` your file structure will looks like:

```Your-App-And-Semantic-Plugins
MyAppSource
│
└───MyPluginsDirectory
    │
    └─── TestPluginFlex
        │
        └─── SloganMakerFlex
        |    |
        │    └─── skprompt.txt
        │    └─── config.json
        │   
        └─── SummarizeBlurbFlex
             |
             └─── skprompt.txt
             └─── config.json
```

When running the kernel in C# you will:

1. Import your desired semantic function by specifying the root plugins directory and the plugin's name
2. Get ready to pass your semantic function parameters with a `ContextVariables` object 
3. Set the corresponding context variables with `<your context variables>.Set`
4. Select the semantic function to run within the plugin by selecting a function

In code, and assuming you've already instantiated and configured your kernel as `kernel` as described [above](/semantic-kernel/howto/semanticfunctions#get-your-kernel-ready):

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.KernelExtensions;
using Microsoft.SemanticKernel.Orchestration;

// ... instantiate a kernel as kernel

var myPlugin = kernel.ImportSemanticSkillFromDirectory("MyPluginsDirectory", "TestPluginFlex");

var myContext = new ContextVariables(); 
myContext.Set("BUSINESS", "Basketweaving Service"); 
myContext.Set("CITY", "Seattle"); 
myContext.Set("SPECIALTY","ribbons"); 

var myResult = await kernel.RunAsync(myContext,myPlugin["SloganMakerFlex"]);

Console.WriteLine(myResult);
```

The output will read similar to:

`"Ribbons with Seattle Style: Quality You Can Count On!"`

## Invoking a semantic function inline from C#

It's possible to bypass the need to package your semantic functions explicitly in `skprompt.txt` files by choosing to create them on-the-fly as inline code at runtime. Let's take `summarizeBlurbFlex`:

```summarizeBlurbFlex
Summarize the following text in two sentences or less. 
---Begin Text---
{{$INPUT}}
---End Text---
```

and define the function inline in C# — assuming you've already instantiated and configured your kernel as `kernel` as described [above](/semantic-kernel/howto/semanticfunctions#get-your-kernel-ready):

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.SemanticFunctions;

// ... instantiate a kernel as kernel

string summarizeBlurbFlex = """
Summarize the following text in two sentences or less. 
---Begin Text---
{{$INPUT}}
---End Text---
""";

var myPromptConfig = new PromptTemplateConfig
{
    Description = "Take an input and summarize it super-succinctly.",
    Completion =
    {
        MaxTokens = 1000,
        Temperature = 0.2,
        TopP = 0.5,
    }
};

var myPromptTemplate = new PromptTemplate(
    summarizeBlurbFlex, 
    myPromptConfig, 
    kernel
);

var myFunctionConfig = new SemanticFunctionConfig(myPromptConfig, myPromptTemplate);

var myFunction = kernel.RegisterSemanticFunction(
    "TestPluginFlex", 
    "summarizeBlurbFlex",
    myFunctionConfig);

var myOutput = await kernel.RunAsync("This is my input that will get summarized for me. And when I go off on a tangent it will make it harder. But it will figure out that the only thing to summarize is that this is a text to be summarized. You think?", 
    myFunction);

Console.WriteLine(myOutput);
```

Note that the configuration was given inline to the kernel with a `PromptTemplateConfig` object instead of a _config.json_ file with the maximum number of tokens to use `MaxTokens`, the variability of words it will use as `TopP`, and the amount of randomness to consider in its response with `Temperature`. Keep in mind that when using C# these parameters will be _PascalCased_ (each word is explicitly capitalized in a string) to be consistent with C# conventions, but in the _config.json_ the parameters are _lowercase._  To learn more about these function parameters read how to [configure functions](/semantic-kernel/howto/configuringfunctions).

A more succinct way to make this happen is with default settings across the board:

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.KernelExtensions;
using Microsoft.SemanticKernel.Orchestration;

// ... instantiate a kernel as kernel

string summarizeBlurbFlex = """
Summarize the following text in two sentences or less. 
---Begin Text---
{{$INPUT}}
---End Text---
""";

var mySummarizeFunction = kernel.CreateSemanticFunction(summarizeBlurbFlex, maxTokens: 1000);

var myOutput = await kernel.RunAsync(
    new ContextVariables("This is my input that will get summarized for me. And when I go off on a tangent it will make it harder But it will figure out that the only thing to summarize is that this is a text to be summarized. You think?"),
    mySummarizeFunction);

Console.WriteLine(myOutput);
```

## Links to learn more about prompts

* [Azure OAI Text Generation Tutorial](/ai-builder/azure-openai-textgen)
* [Transparency Note On Azure OAI](/legal/cognitive-services/openai/transparency-note)
* [Mini-Course on Azure OAI](/training/modules/explore-azure-openai/)
* [OpenAI's Best Practices Guide](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)

## Take the next step

You're now ready to take advantage of the _Kernel's_ pipelining capability. 

> [!div class="nextstepaction"]
> [Compose functions to connect them end-to-end](/semantic-kernel/howto/chainingfunctions)

