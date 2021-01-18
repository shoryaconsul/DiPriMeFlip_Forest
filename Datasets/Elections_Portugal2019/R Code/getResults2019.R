library(stringr)
library(rjson)

options(scipen = 999)

#datafile <- "old_overall_results.csv"
datafile <- "overall_results.csv"
votesfile <- "votes.csv"

leadingUrl <- "https://www.legislativas2019.mai.gov.pt/frontend/data/TerritoryResults?territoryKey=LOCAL-"
trailingUrl <- "&electionId=AR&ts="

results <- NULL
results_votes <- NULL

lookupUrl <- c("500000", #Nacional
               "400000", #Açores
               "010000", #Aveiro
               "020000", #Beja
               "030000", #Braga
               "040000", #Bragança
               "050000", #Castelo Branco
               "060000", #Coimbra
               "070000", #Évora
               "080000", #Faro
               "090000", #Guarda
               "100000", #Leiria
               "110000", #Lisboa
               "300000", #Madeira
               "120000", #Portalegre
               "130000", #Porto
               "140000", #Santarém
               "150000", #Setúbal
               "160000", #Viana do Castelo
               "170000", #Vila Real
               "180000") #Viseu

lookupMaxeleitos <- c(226,#Nacional
                      5,  #Açores
                      16, #Aveiro
                      3,  #Beja
                      19, #Braga
                      3,  #Bragança
                      4,  #Castelo Branco
                      9,  #Coimbra
                      3,  #Évora
                      9,  #Faro
                      3,  #Guarda
                      10, #Leiria
                      48, #Lisboa
                      6,  #Madeira
                      2,  #Portalegre
                      40, #Porto
                      9,  #Santarém
                      18, #Setúbal
                      6,  #Viana do Castelo
                      5,  #Vila Real
                      8)  #Viseu

systime <- Sys.time()

for(i in 1:length(lookupUrl)) {
  
  json_data <- tryCatch({
    
    suppressWarnings(json_file <- paste0(leadingUrl,lookupUrl[i],trailingUrl))
    suppressWarnings(fromJSON(paste(readLines(json_file), collapse="")))
    
  }, warning = function(war) {
    print(paste0("WARNING: ", war))
  }, error = function(err) {
    print(paste0("ERROR: ", err))
  })
  
  if(typeof(json_data)=="list") {
    
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
                          votersPercentage=votersPercentage, subscribedVoters=subscribedVoters, totalVoters=totalVoters)
    
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
      
      maxeleitos <- lookupMaxeleitos[i]
      
      dta <- json_data$currentResults$resultsParty
      
      res <- data.frame(time=numeric(0),District=character(0),Party=character(0),Percentage=numeric(0),Votes=numeric(0))
      
      for(j in 1:length(dta)) {
        res <- rbind(res,data.frame(time=systime,District=as.character(territoryName),Party=as.character(dta[[j]]$acronym),Mandates=as.numeric(dta[[j]]$mandates),
                                    Percentage=as.numeric(dta[[j]]$percentage),validVotesPercentage=dta[[j]]$validVotesPercentage,Votes=as.numeric(dta[[j]]$votes)))
      }
      
      results_votes <- rbind(results_votes,res)
      
    }
    
    

  }
  
  
}

if(!is.null(results)) {
  
  results$time <- as.character(results$time)
  
  # write.csv(results,file=datafile,row.names=FALSE)
  
  if(file.exists(datafile)) {
    dfile <- read.csv(datafile)
    dfile <- rbind(dfile,results)
    write.csv(dfile,file=datafile,row.names=FALSE)
  } else {
    write.csv(results,file=datafile,row.names=FALSE)
  }
  
} else {
  
  print("No Data!")
}


###

if(!is.null(results_votes)) {
  
  results_votes$time <- as.character(results_votes$time)
  
  # write.csv(results,file=datafile,row.names=FALSE)
  
  if(file.exists(votesfile)) {
    dfile <- read.csv(votesfile)
    dfile <- rbind(dfile,results_votes)
    write.csv(dfile,file=votesfile,row.names=FALSE)
  } else {
    write.csv(results_votes,file=votesfile,row.names=FALSE)
  }
  
} else {
  
  print("No Data!")
}

