# import 
import os 
import asyncio
from dotenv import load_dotenv

from deepagents import create_deep_agent 
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

## def main() 
async def main():
    load_dotenv()

    model = "openai:gpt-5.4-mini"

    ## create mcp client 
    client = MultiServerMCPClient({
        "open-meteo":{
            "transport":"streamable_http",
            "url":"https://open-meteo.caseyjhand.com/mcp"
        },
        "tavily":{
            "transport":"streamable_http",
            "url":"https://mcp.tavily.com/mcp/",
            "headers":{
                "Authorization":f"Bearer {os.getenv("TAVILY_API_KEY")}"
            }
        } 
    })

    ## get mcp tools 
    weather_tools, search_tools = await asyncio.gather(
        client.get_tools(server_name="open-meteo"),
        client.get_tools(server_name="tavily")
    )

    if not weather_tools:
        raise RuntimeError("weather tools are not exposed")
    if not search_tools:
            raise RuntimeError("search tools are not exposed")
        
    # print("weather MCP tools: ",", ".join(tool.name for tool in weather_tools))
    # print("\nsearch MCP tools: ",", ".join(tool.name for tool in search_tools))
    
    ## create search_sub_agent using search tools
    search_sub_agent = {
         "name":"search sub agent",
         "description":"web search using tools",
         "system_prompt":"""use tavily mcp search tool to find the relevant context online""",
         "tools":search_tools,
         "model":"openai:gpt-5.4-nano"
    }

    ## create weather_sub_agent using weather tools
    weather_sub_agent = {
             "name":"weather sub agent",
             "description":"weather forcast mcp using tools",
             "system_prompt":"""use open-meteo mcp weather forcast tool to find the relevant context online""",
             "tools":weather_tools,
             "model":"openai:gpt-5.4-nano"
        }
    ## create deep agent 
    agent = create_deep_agent(
         model=model,
         subagents=[search_sub_agent,weather_sub_agent],
         system_prompt="""Delete works to relevant sub agent and synthesize them when they return.
         Otherwise answer of your own.,
         """ 
    )
    while True: 
         user_input = input("User: ").strip()
         
         if user_input.lower() in ["exit","bye","quiz"]:
            break 
         if not user_input:
            continue
         try:
             result = await agent.ainvoke({
              "messages":[HumanMessage(content=user_input)]
         })
             for m in result["messages"]:
                 m.pretty_print()
             #print(result["messages"][-1].text)
         except Exception as e:
            print(f"Invoke error: {e}")

        
        
    
         
         
              
              
         

## invoke main()
if __name__ == "__main__":
    asyncio.run(main())