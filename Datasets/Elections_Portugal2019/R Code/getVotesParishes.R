library(stringr)
library(rjson)

freguesias <- read.csv("freguesias-metadata.csv")
freguesias$brasao <- NULL

options(scipen = 999)

#datafile <- "old_overall_results.csv"
datafile <- "parish_data.csv"

leadingUrl <- "https://www.legislativas2019.mai.gov.pt/frontend/data/TerritoryResults?territoryKey=LOCAL-"
trailingUrl <- "&electionId=AR&ts="

results <- NULL
results_votes <- NULL

systime <- Sys.time()

for(i in 1:nrow(freguesias)) {
  
  print(i)
  
  json_data <- tryCatch({
    
    suppressWarnings(json_file <- paste0(leadingUrl,freguesias[i,]$dicofre,trailingUrl))
    suppressWarnings(fromJSON(paste(readLines(json_file), collapse="")))
    
  }, warning = function(war) {
    print(paste0("WARNING: ", war))
  }, error = function(err) {
    print(paste0("ERROR: ", err))
  })
  
  if(typeof(json_data)=="list") {
    
    if(is.null(json_data$exceptionType)) { # the parish exists
      
      territoryFullName <- json_data$territoryFullName
      territoryName <- json_data$territoryName
      territoryKey <- json_data$territoryKey
      
      dta <- json_data$currentResults
      
      availableMandates <- dta$availableMandates; if(is.null(availableMandates)) availableMandates <- 0
      blankVotes <- dta$blankVotes; if(is.null(blankVotes)) blankVotes <- NA
      blankVotesPercentage <- dta$blankVotesPercentage; if(is.null(blankVotesPercentage)) blankVotesPercentage <- NA
      nullVotes <- dta$nullVotes; if(is.null(nullVotes)) nullVotes <- NA
      nullVotesPercentage <- dta$nullVotesPercentage; if(is.null(nullVotesPercentage)) nullVotesPercentage <- NA
      #voters <- dta$numberVoters; if(is.null(voters)) voters <- NA
      votersPercentage <- dta$percentageVoters; if(is.null(votersPercentage)) votersPercentage <- NA
      subscribedVoters <- dta$subscribedVoters; if(is.null(subscribedVoters)) subscribedVoters <- NA
      totalVoters <- dta$totalVoters; if(is.null(totalVoters)) totalVoters <- NA
      totalMandates <- dta$totalMandates; if(is.null(totalMandates)) totalMandates <- NA
      numParishes <- dta$numberParishes; if(is.null(numParishes)) numParishes <- NA
      numParishesApproved <- dta$totalParishesApproved; if(is.null(numParishesApproved)) numParishesApproved <- NA
      
      new_tbl <- data.frame(time=systime, territoryFullName=territoryFullName, territoryName=territoryName, territoryKey=territoryKey,
                            totalMandates=totalMandates, availableMandates=availableMandates,
                            numParishes=numParishes, numParishesApproved=numParishesApproved,
                            blankVotes=blankVotes, blankVotesPercentage=blankVotesPercentage,
                            nullVotes=nullVotes, nullVotesPercentage=nullVotesPercentage,
                            #voters=voters, 
                            votersPercentage=votersPercentage, subscribedVoters=subscribedVoters) #, totalVoters=totalVoters)
      
      dto <- json_data$previousResults
      
      availableMandates <- dto$availableMandates; if(is.null(availableMandates)) availableMandates <- 0
      blankVotes <- dto$blankVotes; if(is.null(blankVotes)) blankVotes <- NA
      blankVotesPercentage <- dto$blankVotesPercentage; if(is.null(blankVotesPercentage)) blankVotesPercentage <- NA
      nullVotes <- dto$nullVotes; if(is.null(nullVotes)) nullVotes <- NA
      nullVotesPercentage <- dto$nullVotesPercentage; if(is.null(nullVotesPercentage)) nullVotesPercentage <- NA
      #voters <- dto$numberVoters; if(is.null(voters)) voters <- NA
      votersPercentage <- dto$percentageVoters; if(is.null(votersPercentage)) votersPercentage <- NA
      subscribedVoters <- dto$subscribedVoters; if(is.null(subscribedVoters)) subscribedVoters <- NA
      totalVoters <- dto$totalVoters; if(is.null(totalVoters)) totalVoters <- NA
      totalMandates <- dto$totalMandates; if(is.null(totalMandates)) totalMandates <- NA
      
      new_tbl <- cbind(new_tbl,data.frame(pre.totalMandates=totalMandates, pre.availableMandates=availableMandates,
                                          pre.blankVotes=blankVotes, pre.blankVotesPercentage=blankVotesPercentage,
                                          pre.nullVotes=nullVotes, pre.nullVotesPercentage=nullVotesPercentage,
                                          #pre.voters=voters, 
                                          pre.votersPercentage=votersPercentage, pre.subscribedVoters=subscribedVoters, pre.totalVoters=totalVoters))
      
      results <- rbind(results, new_tbl)
      
      # Votes and Hondt calculus
      
      if(!is.null(json_data$currentResults)) {
        
        #maxeleitos <- lookupMaxeleitos[i]
        
        dta <- json_data$currentResults$resultsParty
        
        res <- data.frame(ID=numeric(0),District=character(0),Party=character(0),Percentage=numeric(0),validVotesPercentage=numeric(0),Votes=numeric(0))
        
        for(j in 1:length(dta)) {
          res <- rbind(res,data.frame(ID=i, District=as.character(territoryName),Party=as.character(dta[[j]]$acronym),
                                      Percentage=as.numeric(dta[[j]]$percentage),validVotesPercentage=dta[[j]]$validVotesPercentage,Votes=as.numeric(dta[[j]]$votes)))
        }
        
        results_votes <- rbind(results_votes,res)
        
      }
      
    } else {
      
      # the parish doesn't exist
      
    }
    
  } else {
    stop("Heere at the wall")
  }
  
  if((i %% 25)==0) {
    print("Breathe")
    Sys.sleep(0.5)
  }
  
}

freguesias <- read.csv("freguesias-metadata.csv")

results["Council"] <- freguesias$concelho
results["District"] <- freguesias$District
results$Council <- as.character(results$Council); results$District <- as.character(results$District)
results[results$District=="Ilha da Madeira",]$District <- "Madeira"
results[results$District=="Ilha de Porto Santo",]$District <- "Madeira"
results[grepl("Ilha d",results$District),]$District <- "Açores"

results_votes["Council"] <- results[match(results_votes$District,results$territoryName),]$Council
results_votes["District"] <- results[match(results_votes$District,results$territoryName),]$District
results_votes$ID <- NULL

write.csv(results,file="SourceData/parishes.csv",row.names = FALSE)
write.csv(results_votes,file="SourceData/votes_parishes.csv",row.names = FALSE)


