from model import predict_sentiment
import yaml
from pathlib import Path    


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config/config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sample_text = """
    'Title: Rakuten plays 5G cloud-native Symphony with Robin.ioContent: Robin.io, 
    the Kubernetes storage startup with a 5G edge infrastructure operations and management software stack,
    has been snapped up by Rakuten Symphony.\nRakuten, regarded as Japans version of Amazon, 
    is an online retail company that owns mobile carrier Rakuten Symphony, 
    originally Rakuten Mobile. The Symphony business unit was spun off in August 2021 and has both 4G/5G technology and services in its portfolio.
    Rakuten Symphony CEO Tareq Amin said in a statement: We plan to continue to invest into Robin.ios cloud-native portfolio of products to further 
    advance our capabilities and offer the most advanced and highly integrated cloud platform mobile operators demand.\nEdge cloud requirements are unique and critical as mobile operators transition to 5G.
    The next era of digital experience requires another level of performance,
    responsiveness and consistency that enables telecom operator and enterprise transformation to be safely accelerated while creating a platform to support the next 10 years of experiences.\nRobin.io diagram\nRobin.io was founded in 2013 by CEO Partha Seetala and has raised $86m in venture capital funding. It has more than 70 patents for its cloud-native storageand associated technologies, and started collaborating with Rakuten in 2019 when Rakuten Mobile used Robin.io in production for the Japanese deployment of the worlds first end-to-end fully virtualised cloud-native mobile network. Robin.io has petabyte-size deployments with its software running on tens of thousands of servers in production.\nRakuten said that the addition of Robin.ios multi-cloud mobility, hyper automation and orchestration capabilities to the Rakuten Symphony portfolio allows the creation of highly efficient, consistent high performance cloud infrastructure and operations, from edge to central data centre. The new business unit wants to accelerate and strengthen the complete end-to-end automated cloud offering for customers across the globe.\nSeetala, who will become president of Rakuten Symphonys Unified Cloud business unit, said: I am delighted that Robin.ios technology innovations over the last several years will now get a much bigger canvas to lead the vision for cloud-native transformation for the industry. Our vision to deliver simple to use, easy to deploy hyperscale automation is very well aligned.\nThe acquisition terms are confidential and the deal is subject to closing conditions. As Robin.io is apparently doing extremely well and has unique cloud-native, 5G edge automation technology, we might expect a 4x or 5x return for its investors.\nTAGSTarget Organization: Symphony Corporation'
    """
    predict_sentiment(sample_text, cfg)