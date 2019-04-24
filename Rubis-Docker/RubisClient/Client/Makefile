where := $(lastword $(MAKEFILE_LIST))

include ./config.mk

####################
#       Client     #
####################

ClientFiles = URLGenerator URLGeneratorEJB URLGeneratorServlets URLGeneratorPHP RUBiSProperties Stats \
	      TransitionTable ClientEmulator UserSession InitDB
	    
BeansFiles = TimeManagement	    

all_client_sources =  $(addprefix edu/rice/rubis/client/, $(addsuffix .java, $(ClientFiles))) $(addprefix edu/rice/rubis/beans/, $(addsuffix .java, $(BeansFiles)))
all_client_obj = $(addprefix edu/rice/rubis/client/, $(addsuffix .class, $(ClientFiles))) $(addprefix edu/rice/rubis/beans/, $(addsuffix .class, $(BeansFiles)))

client: $(all_client_obj)
	${JAR} cvf rubis_client.jar $(all_client_obj)

all: client  ## compile the client.

initDB: client ## Initialize the RubisDB.
	${JAVA} -classpath .:./database:rubis_client.jar edu.rice.rubis.client.InitDB ${PARAM}

emulator: client ## Start benchmarking.
	${JAVA} -classpath .:rubis_client.jar edu.rice.rubis.client.ClientEmulator

javadoc:
	${MKDIR} -p doc
	${JAVADOC} -d ./doc -bootclasspath ${CLASSPATH} -version -author -windowtitle "RUBiS Client API" -header "<b>RUBiS (C)2001 Rice University/INRIA</b><br>"  edu.rice.rubis.client

clean:  ## clean up all the mess removing .class.
	${RM} -Rf edu/rice/rubis/client/*.class edu/rice/rubis/beans/*.class doc/* rubis_client.jar

benchmark : initDB emulator ## Compile the source, Init the DB and run the benchmark.
	echo "Benchmarking has been successfully executed"
	
