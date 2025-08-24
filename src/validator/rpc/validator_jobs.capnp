@0xf2a3b4c5d6e7f8a9;

# Schema for validator job distribution RPC

struct EvaluationJobData {
  jobId @0 :UInt64;
  submissionId @1 :UInt64;
  minerHotkey @2 :Text;
  hfRepoId @3 :Text;
  hfRepoCommit @4 :Text;
  envProvider @5 :Text;
  envName @6 :Text;
  logsPath @7 :Text;
  randomSeed @8 :UInt32;
  maxRetries @9 :UInt8;
  retryCount @10 :UInt8;
  createdAt @11 :UInt64; # Unix timestamp in milliseconds
}

# struct for response when registering child
struct RegisterChildResponse {
  success @0 :Bool;
  message @1 :Text;
}

# struct for response when requesting reciving jobs
struct JobResponse {
  accepted @0 :Bool;
  message @1 :Text;
}

interface ValidatorJobsService {
  # Child validator calls this to register with parent
  registerChild @0 (childId: Text, endpoint: Text) -> (response: RegisterChildResponse);
  
  # Child validator calls this to request jobs
  requestJobs @1 (childId: Text, maxJobs: UInt32) -> (jobs: List(EvaluationJobData));
  
  # Child validator calls this to report job completion
  reportJobCompletion @2 (childId: Text, jobId: UInt64, success: Bool, result: Text) -> (acknowledged: Bool);
  
  # Parent calls this on child to send a job directly
  receiveJob @3 (job: EvaluationJobData) -> (accepted: Bool, message: Text);
  
  # Health check
  ping @4 () -> (pong: Text);
}