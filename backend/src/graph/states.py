import operator
from typing import Annotated, List, Dict, Optional, Any, TypedDict

# define the schema for a single compliance issue

class ComplianceIssue(TypedDict):
    category : str
    description : str # specfic detail of violation
    severity : str # CRITICAL, WARNING
    timestamp : Optional[str]

class VideoAuditState(TypedDict):

    '''
    Defines the data schema for langgraph execution content
    Main Container : holds all the information about the audit right from the intial URL to the final report
    '''

    #input parameters
    video_url : str
    video_id : str

    # ingestion and extraction data
    local_file_path : Optional[str]
    video_metadata : Dict[str,Any] # {'duration' : 15, "resolution" : "1080p"}
    transcript : Optional[str] # Fully extracted speech
    ocr_text : List[str] 

    # analysis output
    compliance_results : Annotated[List[ComplianceIssue], operator.add]

    # final deliverables:
    final_status : str # PASS | FAIL
    final_report : str # markdown format

    # system observability
    # errors : API timeout, system level errors
    # list of system level crashes

    errors : Annotated[List[str], operator.add]
    





        